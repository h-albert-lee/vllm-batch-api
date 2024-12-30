from setuptools import setup, find_packages

setup(
    name="vllm_batch_api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "aiohttp",
        "torch",
        "prometheus_client",
        "tqdm",
        "vllm",  # vLLM 라이브러리 의존성
        "pydantic",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "vllm-batch-api = server:main",
        ],
    },
)
