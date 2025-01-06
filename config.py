from pydantic import BaseModel, Field
from typing import Optional
import yaml


class ModelConfig(BaseModel):
    path: str
    tensor_parallel_size: int = Field(default=1, ge=1, description="Number of GPUs for tensor parallelism")


class ServingConfig(BaseModel):
    batch_size: int = Field(default=10, ge=1, description="Number of requests per batch")
    batch_interval: float = Field(default=5.0, ge=0.0, description="Maximum interval (seconds) to wait before processing a batch")
    max_concurrent_requests: int = Field(default=100, ge=1, description="Maximum number of concurrent batch requests")
    response_role: Optional[str] = Field(default="assistant", description="Role name for responses")
    enable_prompt_tokens_details: bool = Field(default=False, description="Enable detailed prompt tokens information")


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO", description="Logging level")
    max_log_len: Optional[int] = Field(default=None, ge=1, description="Maximum length of logs")


class SecurityConfig(BaseModel):
    api_key: str = Field(..., description="API key for authentication")


class BatchConfig(BaseModel):
    model: ModelConfig
    serving: ServingConfig
    logging: LoggingConfig
    security: SecurityConfig


def load_config(config_path: str = "batch_config.yaml") -> BatchConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return BatchConfig(**config_dict)
