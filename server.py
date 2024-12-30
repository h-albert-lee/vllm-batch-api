import asyncio
from typing import List

from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from prometheus_client import start_http_server, Summary, Counter
from batch_processor import BatchProcessor
from config import load_config, BatchConfig
import threading
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM Batch Processing API", version="0.1.0")

class BatchRequest(BaseModel):
    requests: List[str]  # each request -> JSOM

class BatchResponse(BaseModel):
    responses: List[str]  # each response -> JSON

# Prometheus metric set
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('requests_received_total', 'Total number of batch requests received')
REQUESTS_SUCCESS = Counter('requests_success_total', 'Total number of batch requests succeeded')
REQUESTS_FAILURE = Counter('requests_failure_total', 'Total number of batch requests failed')

# API key authentication (option)
from fastapi.security.api_key import APIKeyHeader

config: BatchConfig = load_config("batch_config.yaml")  # load config
API_KEY = config.security.api_key  # load api key from config
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )
    return api_key

# initialize engine
batch_processor = BatchProcessor(config)

# Prometheus metric server start
def start_metrics_server(port: int = 8001):
    start_http_server(port)

@app.on_event("startup")
def startup_event():
    if config.metrics.enable:
        threading.Thread(target=start_metrics_server, args=(config.metrics.port,), daemon=True).start()
        logger.info(f"Prometheus metrics server started on {config.metrics.url}:{config.metrics.port}.")
    else:
        logger.info("Prometheus metrics are disabled.")

@app.post("/batch", response_model=BatchResponse, dependencies=[Depends(get_api_key)])
@REQUEST_TIME.time()
async def handle_batch(request: BatchRequest):
    REQUEST_COUNT.inc()
    logger.info(f"Received batch request with {len(request.requests)} requests.")
    try:
        input_data = "\n".join(request.requests)
        responses = await batch_processor.enqueue_requests(input_data)
        # filer -> metric
        for resp in responses:
            if resp.error is None:
                REQUESTS_SUCCESS.inc()
            else:
                REQUESTS_FAILURE.inc()
        response_json = [resp.model_dump_json() for resp in responses]
        logger.info(f"Batch request processed successfully with {len(responses)} responses.")
        return BatchResponse(responses=response_json)
    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        REQUESTS_FAILURE.inc()
        raise HTTPException(status_code=500, detail=str(e))
