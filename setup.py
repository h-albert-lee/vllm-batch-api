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
        "tqdm",
        "vllm",
        "pydantic",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "vllm-batch-api = server:main",
        ],
    },
)
