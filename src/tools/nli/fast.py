import requests
import aiohttp
import time
from tenacity import retry, stop_after_delay, wait_random_exponential

from cr_utils import CostManagers


def nli(premise: str, hypothesis: str) -> bool:
    params = {"premise": premise, "hypothesis": hypothesis}
    start = time.time()
    rsp = requests.post("http://localhost:8004/nli/", json=params, proxies=None)
    rsp_time = time.time() - start
    CostManagers().update_cost(0, 0, "tool_nli", rsp_time, "tool_nli")
    return rsp.json()

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_delay(120))
async def anli(premise: str, hypothesis: str) -> bool:
    params = {"premise": premise, "hypothesis": hypothesis}
    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8004/nli/", json=params) as response:
            rsp = await response.json()
            rsp_time = time.time() - start
            CostManagers().update_cost(0, 0, "tool_nli", rsp_time, "tool_nli")
            return rsp
