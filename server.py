import asyncio
from typing import List

from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from batch_processor import BatchProcessor
from config import load_config, BatchConfig
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM Batch Processing API", version="0.1.0")

class IndividualRequest(BaseModel):
    custom_id: str
    method: str
    url: str
    body: dict

class BatchRequest(BaseModel):
    requests: List[IndividualRequest]  # 각 요청은 객체 형태로 전달

class BatchResponse(BaseModel):
    responses: List[str]  # 각 응답은 JSON 문자열 형태로 반환


# API 키 인증 (선택 사항)
from fastapi.security.api_key import APIKeyHeader

config: BatchConfig = load_config("batch_config.yaml")
API_KEY = config.security.api_key
API_KEY_NAME = "access_token"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return api_key

# 엔진 초기화
batch_processor = BatchProcessor(config)

@app.post("/batch", response_model=BatchResponse, dependencies=[Depends(get_api_key)])
async def handle_batch(request: BatchRequest):
    logger.info(f"Received batch request with {len(request.requests)} requests.")
    try:
        # 각 개별 요청을 JSON 문자열로 직렬화하여 합친 뒤, enqueue_requests로 전달
        input_data = "\n".join([req.json() for req in request.requests])
        responses = await batch_processor.enqueue_requests(input_data)
        response_json = [r.model_dump_json() for r in responses]
        logger.info(f"Batch request processed successfully with {len(responses)} responses.")
        return BatchResponse(responses=response_json)
    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
