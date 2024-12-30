import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

API_KEY = "your-secure-api-key"

def test_batch_endpoint_success():
    response = client.post(
        "/batch",
        headers={"access_token": API_KEY},
        json={
            "requests": [
                "{\"url\": \"/v1/chat/completions\", \"body\": {\"prompt\": \"Hello, world!\"}}",
                "{\"url\": \"/v1/embeddings\", \"body\": {\"text\": \"OpenAI is awesome.\"}}"
            ]
        }
    )
    assert response.status_code == 200
    assert "responses" in response.json()
    assert len(response.json()["responses"]) == 2
    for resp in response.json()["responses"]:
        assert "id" in resp
        assert "response" in resp
        assert "error" in resp

def test_batch_endpoint_invalid_api_key():
    response = client.post(
        "/batch",
        headers={"access_token": "invalid-key"},
        json={
            "requests": [
                "{\"url\": \"/v1/chat/completions\", \"body\": {\"prompt\": \"Hello, world!\"}}"
            ]
        }
    )
    assert response.status_code == 403

def test_batch_endpoint_invalid_json():
    response = client.post(
        "/batch",
        headers={"access_token": API_KEY},
        json={
            "requests": [
                "{\"url\": \"/v1/chat/completions\", \"body\": {\"prompt\": \"Hello, world!\"}}",
                "invalid-json"
            ]
        }
    )
    assert response.status_code == 200
    responses = response.json()["responses"]
    assert len(responses) == 2
    assert responses[0].get("error") is None
    assert responses[1].get("error") is not None
