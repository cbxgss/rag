import os
import requests
import aiohttp
import time
from tenacity import retry, stop_never, wait_random_exponential
from cr_utils import CostManagers


@retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
def retrieve(source: str, query: str, topk: int = 10):
    data = {
        "source": source,
        "query": query,
        "topk": topk,
    }
    start = time.time()
    session = requests.Session()
    session.trust_env = False
    results = session.post(f"{os.getenv('fastapi_retrieve')}/retrieve/", json=data).json()
    docs, scores, idxs = results["docs"], results["scores"], results["idxs"]
    rsp_time = time.time() - start
    CostManagers().update_cost(0, 0, "tool_retrieve", rsp_time, "tool_retrieve")
    return docs, scores, idxs


@retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
async def aretrieve(source: str, query: str, topk: int = 10):
    async with aiohttp.ClientSession() as session:
        data = {
            "source": source,
            "query": query,
            "topk": topk,
        }
        start = time.time()
        async with session.post(f"{os.getenv('fastapi_retrieve')}/retrieve/", json=data) as response:
            results = await response.json()
            docs, scores, idxs = results["docs"], results["scores"], results["idxs"]
            rsp_time = time.time() - start
            CostManagers().update_cost(0, 0, "tool_retrieve", rsp_time, "tool_retrieve")
            return docs, scores, idxs
