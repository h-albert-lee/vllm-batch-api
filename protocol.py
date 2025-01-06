from pydantic import BaseModel, Field
from typing import Optional, Union, Any, List

class OpenAIBaseModel(BaseModel):
    class Config:
        orm_mode = True


# ----- Request models -----
class ChatCompletionRequest(OpenAIBaseModel):
    model: str
    messages: List[dict]  # [{"role": "...", "content": "..."}]


class EmbeddingRequest(OpenAIBaseModel):
    model: str
    input: Union[str, List[str]]


# ----- Response models -----
class ChatCompletionResponse(OpenAIBaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: Optional[dict] = None


class EmbeddingResponse(OpenAIBaseModel):
    object: str
    data: List[dict]
    model: str


class ErrorResponse(OpenAIBaseModel):
    code: Optional[int] = None
    message: Optional[str] = None
    type: Optional[str] = None
    param: Optional[str] = None
    code_details: Optional[Any] = None


# ----- Batch models -----
class BatchRequestInput(OpenAIBaseModel):
    custom_id: str
    method: str
    url: str
    body: Union[ChatCompletionRequest, EmbeddingRequest]


class BatchResponseData(OpenAIBaseModel):
    status_code: int = 200
    request_id: str
    body: Optional[Union[ChatCompletionResponse, EmbeddingResponse]] = None


class BatchRequestOutput(OpenAIBaseModel):
    id: str
    custom_id: str
    response: Optional[BatchResponseData] = None
    error: Optional[Any] = None
