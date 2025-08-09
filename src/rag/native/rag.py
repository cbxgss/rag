generate_system = """You are a RAG chatbot that will answer questions based on the relevant documents provided.
Please output the answer in a few words, keeping it as brief as possible.
For some questions, simply respond with "yes" or "no."
"""

generate_user = """Question:

{query}

Relevant documents:

{corpus_str}
"""


from omegaconf import DictConfig
import logging
from cr_utils import Logger, Chater

from src.dataset import Item
from src.tools import aretrieve, arerank
from src.rag.base import QA
from src.startup import RAGBuilder


log = logging.getLogger(__name__)


@RAGBuilder.register_module("native")
class NativeRAG(QA):
    def __init__(self, cfg: DictConfig, logger: Logger):
        super().__init__(cfg, logger)

    async def aquery(self, data: Item):
        log_dir = f"log/{data.id}/llm"
        trace = {}
        docs, scores, idxs = await aretrieve(self.cfg.corpus, data.question, self.cfg.task.method.retrieve.topk)
        trace["retrieve"] = {
            "docs": {idx: doc for idx, doc in zip(idxs, docs)},
            "scores": {idx: score for idx, score in zip(idxs, scores)},
        }
        docs, scores, idxs = await arerank(data.question, idxs, docs, k=self.cfg.task.method.retrieve.rerank_topk, filter=self.cfg.task.method.retrieve.rerank_filter)
        response = await self.agenerate(data.question, docs, log_dir)
        trace["rerank"] = {
            "docs": {idx: doc for idx, doc in zip(idxs, docs)},
            "scores": {idx: score for idx, score in zip(idxs, scores)},
        }
        return response, trace

    async def agenerate(self, question: str, docs: str, path):
        corpus_str = docs if isinstance(docs, str) else "\n\n".join(docs)
        messages = [
            {"role": "system", "content": generate_system},
            {"role": "user", "content": generate_user.format(query=question, corpus_str=corpus_str)},
        ]
        rsp = await Chater().acall_llm(messages, model=self.cfg.task.base_llm, name="generate", path=path, temperature=0.3)
        return rsp
