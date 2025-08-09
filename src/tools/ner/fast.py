import requests
import aiohttp
import time
from tenacity import retry, stop_after_delay, wait_random_exponential

from cr_utils import CostManagers


@retry(stop=stop_after_delay(10), wait=wait_random_exponential(multiplier=1, min=1, max=10))
def ner(text: str) -> list[str]:
    params = {"text": text}
    start = time.time()
    rsp = requests.post("http://localhost:8003/ner/", json=params, proxies=None).json()
    rsp = list(dict.fromkeys(rsp))
    rsp_time = time.time() - start
    CostManagers().update_cost(0, 0, "tool_ner", rsp_time, "tool_ner")
    return rsp


@retry(stop=stop_after_delay(10), wait=wait_random_exponential(multiplier=1, min=1, max=10))
async def aner(text: str) -> list[str]:
    params = {"text": text}
    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8003/ner/", json=params) as response:
            rsp = await response.json()
            rsp = list(dict.fromkeys(rsp))
            rsp_time = time.time() - start
            CostManagers().update_cost(0, 0, "tool_ner", rsp_time, "tool_ner")
            return rsp
