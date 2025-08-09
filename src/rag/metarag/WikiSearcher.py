from omegaconf import DictConfig

from src.tools import aretrieve, arerank


class WikiSearcher:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    async def asearch(self, query: str, num1: int, num2: int):
        trace = {}
        docs, scores, idxs = await aretrieve(self.cfg.corpus, query, num1)
        trace["retrieve"] = {
            "docs": {idx: doc for idx, doc in zip(idxs, docs)},
            "scores": {idx: score for idx, score in zip(idxs, scores)},
        }
        docs, scores, idxs = await arerank(query, idxs, docs, k=num2, filter=False)
        trace["rerank"] = {
            "docs": {idx: doc for idx, doc in zip(idxs, docs)},
            "scores": {idx: score for idx, score in zip(idxs, scores)},
        }
        return docs, trace
