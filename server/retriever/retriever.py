import sys
from typing import Callable, Awaitable, TypeVar, ParamSpec
from functools import partial
import dotenv
import time
import pickle as pkl
import numpy as np
import faiss
from pydantic import BaseModel
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from io import BytesIO
import asyncio
from concurrent.futures import ThreadPoolExecutor
sys.path.append("../../")
dotenv.load_dotenv("../../.env")
from src.tools.embedder import FastApiEmbedder

P = ParamSpec('P')
K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")


def make_async(func: Callable[P, T], executor: ThreadPoolExecutor | None) -> Callable[P, Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=executor, func=p_func)

    return _async_wrapper


# cfg_path = "../../config/main.yaml"
# cfg = OmegaConf.load(cfg_path)
# corpus = cfg.corpus
# print(corpus)

app = FastAPI()
gpu = True
embedder = FastApiEmbedder()

class DenseRetriever:
    def __init__(self, embedder: FastApiEmbedder, index_path: str, corpus_path: str, topk=10):
        self.embedder = embedder
        ngpus = faiss.get_num_gpus()
        gpus = range(ngpus)
        print(f"number of gpus: {ngpus}")
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
        res = [faiss.StandardGpuResources() for _ in gpus]
        print(f"read index {index_path}")
        start = time.time()
        self.index = faiss.read_index(index_path)
        print(f"read index time: {time.time() - start:.2f}s")
        if gpu == True:
            print("index to all gpus")
            self.index = faiss.index_cpu_to_gpu_multiple_py(res, self.index, co, gpus)
            print(f"index to all gpus time: {time.time() - start:.2f}s")
        print("read corpus")
        start = time.time()
        self.corpus: list[str] = pkl.load(open(corpus_path, "rb"))
        print(f"read corpus time: {time.time() - start:.2f}s")
        self.topk = topk

    async def retrieve(self, query: str, topk: int = None):
        topk = self.topk if topk is None else topk
        query_emb: np.ndarray = await self.embedder.acreate_embedding([query])
        query_emb = query_emb.astype(np.float32)
        scores, idxs = self.index.search(query_emb, topk)       # faiss 不线程安全, 无法并行
        scores, idxs = scores[0], idxs[0]
        docs = [self.corpus[idx] for idx in idxs]
        return docs, scores.tolist(), idxs.tolist()

corpus_li = ["hotpotqa", "2wikimultihopqa", "musique"]

retrievers = {
    corpus: DenseRetriever(embedder, f"../../tmp/{corpus}.index", f"../../tmp/{corpus}.pkl", topk=10)
    for corpus in corpus_li
}

class RetrieverRequest(BaseModel):
    source: str
    query: str
    topk: int = 10


@app.post("/retrieve/")
async def retrieve(request: Request):
    body = await request.json()
    params = RetrieverRequest(**body)
    print(params.query)
    source = "hotpotqa" if params.source == "wiki" else params.source
    docs, scores, idxs = await retrievers[source].retrieve(params.query, topk=params.topk)
    ret = {
        "idxs": idxs,
        "docs": docs,
        "scores": scores,
    }
    return StreamingResponse(
        BytesIO(json.dumps(ret).encode('utf-8')),
        media_type="application/json",
    )


@app.get("/corpus_len/")
async def corpus_len():
    corpus2len = {
        k: len(v.corpus)
        for k, v in retrievers.items()
    }
    return corpus2len
