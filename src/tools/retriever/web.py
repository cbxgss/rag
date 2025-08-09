import os
import requests
import aiohttp
import time
import json
import dotenv

dotenv.load_dotenv(".env")

url = os.getenv("web_retrieve_api_base", "")
api_key = os.getenv("web_retrieve_api_key", "")

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}


def retrieve(query: str):
    payload = json.dumps({
        "query": query,
        "summary": True,
        "count": 10,
        "page": 1
    })
    start = time.time()
    session = requests.Session()
    session.trust_env = False
    results = session.post(url, headers=headers, data=payload).json()
    breakpoint()
    docs, scores, idxs = results["docs"], results["scores"], results["idxs"]
    rsp_time = time.time() - start
    return docs, scores, idxs

if __name__ == "__main__":
    docs, scores, idxs = retrieve("qwq?")
    print(docs, scores, idxs)
