import requests
import json
import argparse
import sys
import uuid

def create_chat_completion_request(model, messages):
    return {
        "custom_id": str(uuid.uuid4()),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Send batch requests to /batch endpoint for /v1/chat/completions.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of individual requests per batch (1-32).")
    parser.add_argument("--total-requests", type=int, default=10, help="Total number of individual requests to send.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name for chat completions.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/batch", help="Batch endpoint URL.")
    parser.add_argument("--api-key", type=str, required=True, help="API key for authentication.")
    parser.add_argument("--message", type=str, default="Hello, world!", help="User message content.")
    
    args = parser.parse_args()

    if not (1 <= args.batch_size <= 32):
        print("Batch size must be between 1 and 32.")
        sys.exit(1)
    
    if args.total_requests < 1:
        print("Total requests must be at least 1.")
        sys.exit(1)
    
    headers = {
        "Content-Type": "application/json",
        "access_token": args.api_key
    }

    total_batches = (args.total_requests + args.batch_size - 1) // args.batch_size
    sent_requests = 0

    for batch_num in range(1, total_batches + 1):
        current_batch_size = min(args.batch_size, args.total_requests - sent_requests)
        requests_list = []
        for _ in range(current_batch_size):
            individual_request = create_chat_completion_request(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": args.message}
                ]
            )
            requests_list.append(individual_request)
        
        payload = {"requests": requests_list}

        print(f"Sending batch {batch_num}/{total_batches} with {current_batch_size} requests...")
        response = requests.post(args.api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            print(f"Batch {batch_num} processed successfully.")
            resp_data = response.json().get("responses", [])
            for i, resp_str in enumerate(resp_data, start=1):
                try:
                    resp_json = json.loads(resp_str)
                    print(f"  Response {i}:")
                    print(json.dumps(resp_json, indent=2))
                except json.JSONDecodeError:
                    print(f"  Response {i}: {resp_str}")
        else:
            print(f"Batch {batch_num} failed with status code {response.status_code}.")
            print(response.text)
        
        sent_requests += current_batch_size

if __name__ == "__main__":
    main()
