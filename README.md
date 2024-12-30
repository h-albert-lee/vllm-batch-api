# vLLM Batch API Server

**vLLM Batch API Server** is a scalable and efficient server built on top of the `vllm run_batch.py` script. It leverages FastAPI to provide a real-time batch processing API, optimized for multi-GPU environments. Designed for high throughput and low latency, it seamlessly integrates with Prometheus for monitoring and includes robust logging and security features.

## Features

- **Real-time Batch Processing**: Handle multiple requests concurrently with efficient batching.
- **Multi-GPU Support**: Utilize tensor parallelism across multiple GPUs for enhanced performance.
- **Configurable Settings**: Manage server configurations via a YAML file (`batch_config.yaml`).
- **Prometheus Integration**: Monitor server metrics for performance and reliability.
- **Secure Access**: API key-based authentication to secure your endpoints.
- **Structured Logging**: Comprehensive logging for easy debugging and monitoring.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-repo/vllm-batch-api.git
    cd vllm-batch-api
    ```

2. **Install Dependencies**

    Ensure you have Python 3.8 or higher installed. Then, install the required packages:

    ```bash
    pip install .
    ```

## Configuration

Configure the server by editing the `batch_config.yaml` file. Below is an example configuration:

```yaml
model:
  path: "path/to/your/model"
  tensor_parallel_size: 4  # Number of GPUs to use

serving:
  batch_size: 10
  batch_interval: 5.0  # Seconds
  max_concurrent_requests: 100
  response_role: "assistant"
  enable_prompt_tokens_details: true

metrics:
  enable: true
  url: "0.0.0.0"
  port: 8001

logging:
  level: "INFO"
  max_log_len: 1000

security:
  api_key: "your-secure-api-key"
```

## Configuration Parameters
**model.path** : Path to your vLLM model.  
**model.tensor_parallel_size** : Number of GPUs to utilize for tensor parallelism.  
**serving.batch_size** : Number of requests to process in a single batch.  
**serving.batch_interval** : Maximum time (in seconds) to wait before processing a batch.  
**serving.max_concurrent_requests** : Maximum number of concurrent batch requests.  
**serving.response_role** : Role name for responses (e.g., "assistant").  
**serving.enable_prompt_tokens_details** : Enable detailed prompt token tracking.  
**metrics.enable** : Enable Prometheus metrics.  
**metrics.url** : Host for the Prometheus metrics server.  
**metrics.port** : Port for the Prometheus metrics server.  
**logging.level** : Logging level (e.g., "INFO", "DEBUG").  
**logging.max_log_len** : Maximum length of logs.  
**security.api_key** : API key for securing the API endpoints.
  

## Running the Server
Start the FastAPI server using Uvicorn:

    ```bash
    uvicorn server:app --host 0.0.0.0 --port 8000
    ```
Alternatively, if you have set up the console script via setup.py:

    ```bash
    vllm-batch-api
    ```
## API Usage
### Endpoint
POST /batch

### Request

Send a JSON payload containing a list of requests. Each request should be a JSON string compatible with OpenAI's API format.

    ```json
    {
    "requests": [
        "{\"url\": \"/v1/chat/completions\", \"body\": {\"prompt\": \"Hello, world!\"}}",
        "{\"url\": \"/v1/embeddings\", \"body\": {\"text\": \"OpenAI is awesome.\"}}"
    ]
    }
    ```
### Response
Returns a JSON payload containing the list of responses for each request.

    ```json
        {
    {
        "responses": [
            "{\"id\": \"vllm-<uuid>\", \"custom_id\": null, \"response\": {\"body\": {...}, \"request_id\": \"vllm-batch-<uuid>\"}, \"error\": null}",
            "{\"id\": \"vllm-<uuid>\", \"custom_id\": null, \"response\": {\"body\": {...}, \"request_id\": \"vllm-batch-<uuid>\"}, \"error\": null}"
        ]
    }

        }
    ```

### Authentication
Include the API key in the request headers

### Example with curl
    ```bash
    curl -X POST "http://localhost:8000/batch" \
        -H "Content-Type: application/json" \
        -H "access_token: your-secure-api-key" \
        -d '{
            "requests": [
                "{\"url\": \"/v1/chat/completions\", \"body\": {\"prompt\": \"Hello, world!\"}}",
                "{\"url\": \"/v1/embeddings\", \"body\": {\"text\": \"OpenAI is awesome.\"}}"
            ]
            }'
    ```

### Monitoring
Prometheus metrics are available at http://<server_ip>:8001/metrics. Ensure Prometheus is configured to scrape this endpoint to monitor server performance and request statistics

### Logging
The server logs important events and errors. Logs are configured based on the logging section in batch_config.yaml. Adjust the logging level and maximum log length as needed.
