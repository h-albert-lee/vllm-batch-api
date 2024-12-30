import asyncio
from http import HTTPStatus
from typing import Callable, List, Optional, Tuple

import torch
from prometheus_client import Counter, Summary
from tqdm.asyncio import tqdm

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger, logger
from vllm.entrypoints.openai.protocol import (
    BatchRequestInput,
    BatchRequestOutput,
    BatchResponseData,
    ChatCompletionResponse,
    EmbeddingResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import BaseModelPath
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid

from config import BatchConfig

# Prometheus metrics
REQUESTS_RECEIVED = Counter('requests_received_total', 'Total number of batch requests received')
REQUESTS_PROCESSED = Counter('requests_processed_total', 'Total number of batch requests processed')
REQUESTS_FAILED = Counter('requests_failed_total', 'Total number of batch requests failed')
REQUEST_PROCESSING_TIME = Summary('request_processing_seconds', 'Time spent processing requests')


class BatchProgressTracker:
    def __init__(self, total: int):
        self._total = total
        self._pbar: Optional[tqdm] = None

    def submitted(self):
        pass  # No-op since total is predefined

    def completed(self):
        if self._pbar:
            self._pbar.update()

    async def pbar(self):
        enable_tqdm = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        self._pbar = tqdm(
            total=self._total,
            unit="req",
            desc="Running batch",
            mininterval=5,
            disable=not enable_tqdm,
            bar_format="{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n",
        )
        async with self._pbar:
            await asyncio.sleep(0)  # Keeps the progress bar open


def make_error_request_output(request: BatchRequestInput, error_msg: str) -> BatchRequestOutput:
    batch_output = BatchRequestOutput(
        id=f"vllm-{random_uuid()}",
        custom_id=request.custom_id,
        response=BatchResponseData(
            status_code=HTTPStatus.BAD_REQUEST,
            request_id=f"vllm-batch-{random_uuid()}",
        ),
        error=error_msg,
    )
    return batch_output


async def make_async_error_request_output(request: BatchRequestInput, error_msg: str) -> BatchRequestOutput:
    return make_error_request_output(request, error_msg)


async def run_request(serving_engine_func: Callable, request: BatchRequestInput, tracker: BatchProgressTracker) -> BatchRequestOutput:
    try:
        response = await serving_engine_func(request.body)
    except Exception as e:
        logger.error(f"Error processing request {request.id}: {e}")
        return make_error_request_output(request, error_msg=str(e))

    if isinstance(response, (ChatCompletionResponse, EmbeddingResponse)):
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=BatchResponseData(
                body=response, request_id=f"vllm-batch-{random_uuid()}"),
            error=None,
        )
    elif isinstance(response, ErrorResponse):
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=BatchResponseData(
                status_code=response.code,
                request_id=f"vllm-batch-{random_uuid()}"),
            error=response.message if hasattr(response, 'message') else "Unknown error",
        )
    else:
        batch_output = make_error_request_output(
            request, error_msg="Unexpected response type from the serving engine"
        )

    tracker.completed()
    return batch_output


class BatchProcessor:
    def __init__(self, config: BatchConfig):
        self.config = config
        self.batch_size = config.serving.batch_size
        self.batch_interval = config.serving.batch_interval
        self.max_concurrent_requests = config.serving.max_concurrent_requests
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_concurrent_requests)
        self.engine_args = AsyncEngineArgs(
            model=config.model.path,
            tensor_parallel_size=config.model.tensor_parallel_size,
            # Add other necessary arguments from config if needed
        )
        self.engine = AsyncLLMEngine.from_engine_args(
            self.engine_args, usage_context=UsageContext.OPENAI_BATCH_RUNNER
        )
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.initialize_engine())
        self.loop.create_task(self.batch_worker())

    async def initialize_engine(self):
        self.model_config = await self.engine.get_model_config()
        self.served_model_names = [f"{self.config.model.path}-model-{i}" for i in range(self.config.model.tensor_parallel_size)]
        self.base_model_paths = [BaseModelPath(name=name, model_path=self.config.model.path) for name in self.served_model_names]

        if self.config.logging.max_log_len:
            self.request_logger = RequestLogger(max_log_len=self.config.logging.max_log_len)
        else:
            self.request_logger = RequestLogger()

        # Initialize serving objects
        self.openai_serving_chat = (
            OpenAIServingChat(
                self.engine,
                self.model_config,
                self.base_model_paths,
                self.config.serving.response_role,
                lora_modules=None,
                prompt_adapters=None,
                request_logger=self.request_logger,
                chat_template=None,
                chat_template_content_format="auto",
                enable_prompt_tokens_details=self.config.serving.enable_prompt_tokens_details,
            ) if self.model_config.runner_type == "generate" else None
        )
        self.openai_serving_embedding = (
            OpenAIServingEmbedding(
                self.engine,
                self.model_config,
                self.base_model_paths,
                self.request_logger,
                chat_template=None,
                chat_template_content_format="auto",
            ) if self.model_config.task == "embed" else None
        )
        logger.info("vLLM engine initialized and serving objects created.")

    async def enqueue_requests(self, input_data: str) -> List[BatchRequestOutput]:
        # Parse input data into BatchRequestInput objects
        requests = []
        for request_json in input_data.strip().split("\n"):
            request_json = request_json.strip()
            if not request_json:
                continue
            try:
                request = BatchRequestInput.model_validate_json(request_json)
                requests.append(request)
            except Exception as e:
                logger.error(f"Invalid request JSON: {request_json}, error: {e}")
                # Create an error response for invalid JSON
                error_response = make_error_request_output(
                    BatchRequestInput(id="invalid", custom_id=None, url="", body={}),
                    error_msg=f"Invalid request JSON: {e}"
                )
                requests.append(error_response)

        # Enqueue all requests
        response_futures = []
        for request in requests:
            if isinstance(request, BatchRequestOutput):
                # Already an error response
                response_futures.append(asyncio.create_task(asyncio.sleep(0, result=request)))
                REQUESTS_RECEIVED.inc()
                REQUESTS_PROCESSED.inc()
                continue

            future = self.loop.create_future()
            try:
                await self.queue.put((request, future))
                response_futures.append(future)
                REQUESTS_RECEIVED.inc()
            except asyncio.QueueFull:
                logger.warning("Request queue is full. Dropping request.")
                error_response = make_error_request_output(
                    request, error_msg="Server is busy. Please try again later."
                )
                response_futures.append(asyncio.create_task(asyncio.sleep(0, result=error_response)))
                REQUESTS_FAILED.inc()

        return await asyncio.gather(*response_futures)

    async def batch_worker(self):
        while True:
            batch = []
            try:
                # Wait for the first request
                request, future = await self.queue.get()
                batch.append((request, future))
                # Collect up to batch_size - 1 more requests, waiting up to batch_interval seconds
                for _ in range(self.batch_size - 1):
                    try:
                        request, future = await asyncio.wait_for(self.queue.get(), timeout=self.batch_interval)
                        batch.append((request, future))
                    except asyncio.TimeoutError:
                        break  # Time to process the current batch

                # Process the batch
                await self.process_batch(batch)
            except Exception as e:
                logger.error(f"Error in batch_worker: {e}")

    async def process_batch(self, batch: List[Tuple[BatchRequestInput, asyncio.Future]]) -> None:
        tracker = BatchProgressTracker(total=len(batch))
        processing_task = asyncio.create_task(tracker.pbar())

        response_futures = [
            run_request(self.get_handler_fn(request.url), request, tracker)
            for request, _ in batch
        ]

        REQUESTS_PROCESSED.inc(len(batch))

        responses = await asyncio.gather(*response_futures, return_exceptions=True)

        for (request, future), response in zip(batch, responses):
            if isinstance(response, Exception):
                response = make_error_request_output(request, error_msg=str(response))
                REQUESTS_FAILED.inc()
            elif response.error is not None:
                REQUESTS_FAILED.inc()
            future.set_result(response)

        await processing_task  # Ensure progress bar completes
        logger.info(f"Processed batch of {len(batch)} requests.")

    def get_handler_fn(self, url: str) -> Optional[Callable]:
        if url == "/v1/chat/completions":
            return self.openai_serving_chat.create_chat_completion if self.openai_serving_chat else None
        elif url == "/v1/embeddings":
            return self.openai_serving_embedding.create_embedding if self.openai_serving_embedding else None
        else:
            return None
