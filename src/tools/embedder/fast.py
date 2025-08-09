import os
import numpy as np
from openai.types import CreateEmbeddingResponse
import requests
import aiohttp
from tenacity import retry, stop_never, wait_random_exponential


class FastApiEmbedder:
    @staticmethod
    def create_embedding(corpus: list[str], batch_size=1) -> np.ndarray:
        @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
        def ask_embedding(corpus: list[str]):
            response = requests.post(f"{os.getenv('fastapi_embed')}/create_embedding/", json=corpus, proxies=None)
            response = CreateEmbeddingResponse(**response.json())
            return response

        response_embs = []
        for start_idx in range(0, len(corpus), batch_size):
            batch_data = corpus[start_idx:start_idx+batch_size]
            batch_embs: CreateEmbeddingResponse = ask_embedding(batch_data)
            response_embs += [emb.embedding for emb in batch_embs.data]
        response_embs = np.array(response_embs)
        return response_embs

    @staticmethod
    async def acreate_embedding(corpus: list[str], batch_size=1) -> np.ndarray:
        @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
        async def aask_embedding(corpus: list[str]):
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{os.getenv('fastapi_embed')}/create_embedding/", json=corpus) as response:
                    data = await response.json()
                    response = CreateEmbeddingResponse(**data)
                    return response

        response_embs = []
        for start_idx in range(0, len(corpus), batch_size):
            batch_data = corpus[start_idx:start_idx+batch_size]
            batch_embs: CreateEmbeddingResponse = await aask_embedding(batch_data)
            response_embs += [emb.embedding for emb in batch_embs.data]
        response_embs = np.array(response_embs)
        return response_embs

    @staticmethod
    @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
    def get_dim():
        response = requests.get(f"{os.getenv('fastapi_embed')}/get_dim/")
        return response.json()

    @staticmethod
    @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
    def get_max_seq_length():
        response = requests.get(f"{os.getenv('fastapi_embed')}/get_max_seq_length/")
        return response.json()
