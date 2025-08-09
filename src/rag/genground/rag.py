from omegaconf import DictConfig
import logging
from tqdm.asyncio import tqdm as atqdm
import json
import re
from cr_utils import Logger

from tenacity import retry, stop_after_delay, wait_random_exponential
from src.dataset import Item
from src.tools import LLMAgent, aretrieve, arerank
from src.rag.base import QA
from src.rag.genground.prompt import I_A, response_format_A, I_G, I_answer
from src.startup import RAGBuilder


log = logging.getLogger(__name__)


@RAGBuilder.register_module("genground")
class GenGround(QA):
    def __init__(
            self,
            cfg: DictConfig,
            logger: Logger,
    ):
        super().__init__(cfg, logger)
        self.agent1 = LLMAgent("decuce_answer", I_A, cfg.task.method.decuce_answer, response_format_A)
        self.agent2 = LLMAgent("ground", I_G, cfg.task.method.ground)
        self.answer = LLMAgent("answer", I_answer, cfg.task.base_llm)

    async def aquery(self, data: Item):
        try:
            self.logger.mkdir(f"log/{data.id}/llm")
            trace_tmp = {}
            history = []
            with atqdm(range(self.cfg.task.method.max_iter), leave=False, desc=f"{data.id}") as tbar:
                for step in tbar:
                    tbar.set_postfix(status="deduce_answer")
                    q, a, rsp = await self.adeduce_answer(data, history)
                    tbar.set_postfix(status="retrieve")
                    docs, trace_re = await self.aretrieve(q)
                    tbar.set_postfix(status="ground")
                    a_new, ref, rsp = await self.aground(data, q, a, docs)
                    history.append((q, a_new))
                    trace_tmp[step] = {
                        "deduce_answer": {
                            "subquestion": q,
                            "answer": a,
                            "rsp": rsp,
                        },
                        "retrieve": trace_re,
                        "ground": {
                            "answer": a_new,
                            "ref": ref,
                            "rsp": rsp,
                        },
                        "history_step": [q, a, a_new]
                    }
            rsp = await self.answer.arun(data.id, placeholder={
                "question": data.question,
                "history": "\n".join([
                    f"- Item {i}:\n  - subquestion: {subquestion}\n  - answer: {answer}"
                    for i, (subquestion, answer) in enumerate(history)
                ]),
            })
            trace = {
                "answer": rsp,
                "history": {
                    i: trace_step["history_step"]
                    for i, trace_step in trace_tmp.items()
                },
                "trace": trace_tmp,
            }
            return rsp, trace
        except Exception as e:
            log.error(f"Error: {e}")
            return "EMPTY", {}

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_delay(120))
    async def adeduce_answer(self, data: Item, history: list[tuple[str, str]]):
        placeholder = {
            "question": data.question,
            "history": "\n".join([
                f"- Item {i}:\n  - subquestion: {subquestion}\n  - answer: {answer}"
                for i, (subquestion, answer) in enumerate(history)
            ]),
        }
        rsp = await self.agent1.arun(data.id, placeholder)
        rsp = json.loads(rsp)
        subquestion, answer = rsp["subquestion"], rsp["answer"]
        return subquestion, answer, rsp

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_delay(120))
    async def aground(self, data: Item, subquestion: str, answer: str, docs: list[str]):
        bs = self.cfg.task.method.batch
        ref, a_new, rsp = "Empty", answer, ""
        for i in range(0, len(docs), bs):
            docs_b = docs[i: min(i + bs, len(docs))]
            placeholder = {
                "question": data.question,
                "subquestion": subquestion,
                "answer": answer,
                "docs": "\n\n".join(docs_b),
            }
            rsp = await self.agent2.arun(data.id, placeholder)
            ref = re.search(r"<ref>(.*?)</ref>", rsp, re.DOTALL).group(1).strip()
            if ref != "Empty":
                a_new = re.search(r"<revise>(.*?)</revise>", rsp, re.DOTALL).group(1).strip()
                break
        return a_new, ref, rsp

    async def aretrieve(self, query: str):
        trace = {}
        docs, scores, idxs = await aretrieve(self.cfg.corpus, query, self.cfg.task.method.retrieve.topk)
        trace["retrieve"] = {
            "docs": {idx: doc for idx, doc in zip(idxs, docs)},
            "scores": {idx: score for idx, score in zip(idxs, scores)},
        }
        docs, scores, idxs = await arerank(query, idxs, docs, self.cfg.task.method.retrieve.rerank_topk, filter=False)
        trace["rerank"] = {
            "docs": {idx: doc for idx, doc in zip(idxs, docs)},
            "scores": {idx: score for idx, score in zip(idxs, scores)},
        }
        return docs, trace
