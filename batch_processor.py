import asyncio
from http import HTTPStatus
from typing import Callable, List, Optional, Tuple

import torch
from tqdm.asyncio import tqdm

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger, logger
from protocol import (
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


class BatchProgressTracker:
    def __init__(self, total: int):
        self._total = total
        self._pbar: Optional[tqdm] = None

    def submitted(self):
        pass  # No-op, since total is predefined

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
            bar_format=(
                "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]\n"
            ),
        )
        async with self._pbar:
            await asyncio.sleep(0)  # Keeps the progress bar open


def make_error_request_output(
    request: BatchRequestInput, error_msg: str
) -> BatchRequestOutput:
    return BatchRequestOutput(
        id=f"vllm-{random_uuid()}",
        custom_id=request.custom_id,
        response=BatchResponseData(
            status_code=HTTPStatus.BAD_REQUEST,
            request_id=f"vllm-batch-{random_uuid()}",
        ),
        error=error_msg,
    )


async def run_request(
    serving_engine_func: Callable,
    request: BatchRequestInput,
    tracker: BatchProgressTracker,
) -> BatchRequestOutput:
    try:
        # 현재는 POST만 지원
        if request.method.upper() == "POST":
            response = await serving_engine_func(request.body)
        else:
            raise ValueError(f"Unsupported HTTP method: {request.method}")
    except Exception as e:
        logger.error(f"Error processing request {request.custom_id}: {e}")
        return make_error_request_output(request, error_msg=str(e))
    else:
        # try 블록에서 예외가 없으면 이 else 블록이 실행됩니다.
        if isinstance(response, (ChatCompletionResponse, EmbeddingResponse)):
            return BatchRequestOutput(
                id=f"vllm-{random_uuid()}",
                custom_id=request.custom_id,
                response=BatchResponseData(
                    body=response,
                    request_id=f"vllm-batch-{random_uuid()}",
                ),
                error=None,
            )
        elif isinstance(response, ErrorResponse):
            return BatchRequestOutput(
                id=f"vllm-{random_uuid()}",
                custom_id=request.custom_id,
                response=BatchResponseData(
                    status_code=response.code,
                    request_id=f"vllm-batch-{random_uuid()}",
                ),
                error=response.message if hasattr(response, "message") else "Unknown error",
            )
        else:
            return make_error_request_output(
                request, error_msg="Unexpected response type from the serving engine"
            )
    finally:
        # try/except/else 블록을 빠져나가기 전, 항상 실행됩니다.
        tracker.completed()


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
        )
        self.engine = AsyncLLMEngine.from_engine_args(
            self.engine_args, usage_context=UsageContext.OPENAI_BATCH_RUNNER
        )

        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.initialize_engine())
        self.loop.create_task(self.batch_worker())

    async def initialize_engine(self):
        self.model_config = await self.engine.get_model_config()

        # 모델 경로를 기반으로 (tensor_parallel_size만큼) base_model_paths 생성
        self.served_model_names = [
            f"{self.config.model.path}-model-{i}"
            for i in range(self.config.model.tensor_parallel_size)
        ]
        self.base_model_paths = [
            BaseModelPath(name=name, model_path=self.config.model.path)
            for name in self.served_model_names
        ]

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
            )
            if self.model_config.runner_type == "generate"
            else None
        )
        self.openai_serving_embedding = (
            OpenAIServingEmbedding(
                self.engine,
                self.model_config,
                self.base_model_paths,
                self.request_logger,
                chat_template=None,
                chat_template_content_format="auto",
            )
            if self.model_config.task == "embed"
            else None
        )
        logger.info("vLLM engine initialized and serving objects created.")

    async def enqueue_requests(self, input_data: str) -> List[BatchRequestOutput]:
        # 개별 요청 파싱
        requests: List[BatchRequestInput] = []
        for request_json in input_data.strip().split("\n"):
            line = request_json.strip()
            if not line:
                continue
            try:
                request = BatchRequestInput.parse_raw(line)
                # /v1/chat/completions, /v1/embeddings에 필요한 필드를 검증
                if request.url == "/v1/chat/completions":
                    if "model" not in request.body or "messages" not in request.body:
                        raise ValueError("Missing 'model' or 'messages' in body for chat completions.")
                elif request.url == "/v1/embeddings":
                    if "model" not in request.body or "input" not in request.body:
                        raise ValueError("Missing 'model' or 'input' in body for embeddings.")
                requests.append(request)
            except Exception as e:
                logger.error(f"Invalid request JSON: {request_json}, error: {e}")
                # error response 생성
                error_response = make_error_request_output(
                    BatchRequestInput(
                        custom_id="invalid",
                        method="POST",
                        url="",
                        body={},
                    ),
                    error_msg=f"Invalid request: {e}",
                )
                # 이미 BatchRequestOutput 형태이므로, requests 리스트에 그대로 추가
                requests.append(error_response)

        # 요청 큐에 추가
        response_futures = []
        for req in requests:
            if isinstance(req, BatchRequestOutput):
                # 이미 오류가 발생한 요청
                response_futures.append(asyncio.create_task(asyncio.sleep(0, result=req)))
                continue

            future = self.loop.create_future()
            try:
                await self.queue.put((req, future))
            except asyncio.QueueFull:
                logger.warning("Request queue is full. Dropping request.")
                error_response = make_error_request_output(
                    req, error_msg="Server is busy. Please try again later."
                )
                response_futures.append(asyncio.create_task(asyncio.sleep(0, result=error_response)))
                continue

            response_futures.append(future)

        return await asyncio.gather(*response_futures)

    async def batch_worker(self):
        while True:
            batch: List[Tuple[BatchRequestInput, asyncio.Future]] = []
            try:
                # 첫 요청을 기다림
                request, future = await self.queue.get()
                batch.append((request, future))

                # batch_size-1 만큼 추가 요청 대기
                for _ in range(self.batch_size - 1):
                    try:
                        req, fut = await asyncio.wait_for(self.queue.get(), timeout=self.batch_interval)
                        batch.append((req, fut))
                    except asyncio.TimeoutError:
                        break

                await self.process_batch(batch)

            except Exception as e:
                logger.error(f"Error in batch_worker: {e}")

    async def process_batch(self, batch: List[Tuple[BatchRequestInput, asyncio.Future]]) -> None:
        tracker = BatchProgressTracker(total=len(batch))
        pbar_task = asyncio.create_task(tracker.pbar())

        tasks = [
            run_request(self.get_handler_fn(req.url), req, tracker)
            for req, _ in batch
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for (req, fut), resp in zip(batch, responses):
            if isinstance(resp, Exception):
                resp = make_error_request_output(req, error_msg=str(resp))
            fut.set_result(resp)

        await pbar_task  # tqdm 종료
        logger.info(f"Processed batch of {len(batch)} requests.")

    def get_handler_fn(self, url: str) -> Optional[Callable]:
        if url == "/v1/chat/completions":
            return self.openai_serving_chat.create_chat_completion if self.openai_serving_chat else None
        elif url == "/v1/embeddings":
            return self.openai_serving_embedding.create_embedding if self.openai_serving_embedding else None
        else:
            return None
