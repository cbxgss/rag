import os
import requests
import aiohttp
import time
from tenacity import retry, stop_never, wait_random_exponential
from cr_utils import CostManagers


@retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
def rerank(query: str, idxs: list[int], documents: list[str], k: int = 5, filter: bool = True) -> tuple[list[str], list[float], list[int]]:
    if len(idxs) == 0:
        return [], [], []
    data = {
        "query": query,
        "documents": documents,
    }
    start = time.time()
    session = requests.Session()
    session.trust_env = False
    scores = session.post(f"{os.getenv('fastapi_rerank')}/rerank/", json=data).json()
    ranked_docs = sorted(zip(idxs, documents, scores), key=lambda x: x[2], reverse=True)
    if filter:
        ranked_docs = [doc for doc in ranked_docs if doc[2] > 0]
    idxs = [idx for idx, _, _ in ranked_docs[:k]]
    scores = [score for _, _, score in ranked_docs[:k]]
    docs = [doc for _, doc, _ in ranked_docs[:k]]
    rsp_time = time.time() - start
    CostManagers().update_cost(0, 0, "tool_rerank", rsp_time, "tool_rerank")
    return docs, scores, idxs


@retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
async def arerank(query: str, idxs: list[int], documents: list[str], k: int = 5, filter: bool = True) -> tuple[list[str], list[float], list[int]]:
    if len(idxs) == 0:
        return [], [], []
    async with aiohttp.ClientSession() as session:
        data = {
            "query": query,
            "documents": documents,
        }
        start = time.time()
        async with session.post(f"{os.getenv('fastapi_rerank')}/rerank/", json=data) as response:
            scores = await response.json()
            ranked_docs = sorted(zip(idxs, documents, scores), key=lambda x: x[2], reverse=True)
            if filter:
                ranked_docs = [doc for doc in ranked_docs if doc[2] > 0]
            idxs = [idx for idx, _, _ in ranked_docs[:k]]
            scores = [score for _, _, score in ranked_docs[:k]]
            docs = [doc for _, doc, _ in ranked_docs[:k]]
            rsp_time = time.time() - start
            CostManagers().update_cost(0, 0, "tool_rerank", rsp_time, "tool_rerank")
            return docs, scores, idxs
