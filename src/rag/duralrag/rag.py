from copy import deepcopy
from omegaconf import DictConfig
from tqdm.asyncio import tqdm
import json
import logging
from cr_utils import Logger

from tenacity import retry, stop_never, wait_random_exponential
from src.dataset import Item
from src.tools import LLMAgent, aretrieve, arerank
from src.rag.base import QA
from src.rag.duralrag.doc import Doc, Docs
from src.rag.duralrag.prompt import *
from src.startup import RAGBuilder


log = logging.getLogger(__name__)


class Knowledge:
    def __init__(self, entity: str) -> None:
        self.entity: str = entity
        self.contents: list[str] = []
        self.supports: Docs = Docs()

    def __str__(self) -> str:
        return f"#### {self.entity}\n\n" + "\n\n".join(self.contents)

    @classmethod
    def dict2str(cls, knowledge: dict[str, "Knowledge"]) -> str:
        return "\n\n".join([str(k) for k in knowledge.values()])

    @classmethod
    def dict2json(cls, knowledge: dict[str, "Knowledge"]) -> dict:
        return deepcopy({k: v.contents for k, v in knowledge.items()})


class Infer(LLMAgent):
    def __init__(self, cfg: DictConfig) -> None:
        if cfg.task.method.abl.infer:
            super().__init__("infer", prompt_infer_abl, cfg.task.method.infer)
        else:
            super().__init__("infer", prompt_infer, cfg.task.method.infer, response_format_infer)
        self.cfg = cfg

    @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
    async def ainfer(self, id: str, placeholder: dict):
        if self.cfg.task.method.abl.infer:
            rsp = await self.arun(id, placeholder, openai_params={"stop": [".", "\n"], "temperature": 0.3}) + "."
            need_retrieve = not "So the answer is" in rsp
            return rsp, need_retrieve, rsp
        else:
            rsp = await self.arun(id, placeholder)
            rsp = json.loads(rsp)
            thought, need_retrieve = rsp["thought"], rsp["need_retrieve"]
            return thought, need_retrieve, rsp


class KManager:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.kb: dict[str, Knowledge] = {}  # entity -> Knowledge
        self.thought: list[str] = []
        self.agents = {
            "EI": (
                LLMAgent("EI", prompt_need, cfg.task.method.EI, response_format_need)
                if not cfg.task.method.abl.EI2 else
                LLMAgent("EI2", prompt_need_abl2, cfg.task.method.EI, response_format_need_abl2)
            ),
            "KS": LLMAgent("KS", prompt_learn, cfg.task.method.KS),
        }

    @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
    async def aei(self, data: Item):
        if self.cfg.task.method.abl.EI:
            entity2key = {"none": [self.thought[-1]]}
            return entity2key
        elif self.cfg.task.method.abl.EI2:
            rsp = await self.agents["EI"].arun(data.id, placeholder={
                "question": data.question,
                "thought": "\n".join([f"{i + 1}. {t}" for i, t in enumerate(self.thought)]),
            })
            query = json.loads(rsp)["query"]
            return {data.question: [query]}
        else:
            rsp = await self.agents["EI"].arun(data.id, placeholder={
                "knowledge": Knowledge.dict2str(self.kb),
                "question": data.question,
                "thought": "\n".join([f"{i + 1}. {t}" for i, t in enumerate(self.thought)]),
                "known_entity": [entity for entity in self.kb.keys() if entity != "else"],
            })
            entity2key = json.loads(rsp)["entities"]
            entity2key = {entity["entity"]: entity["keywords"] for entity in entity2key}
            return entity2key

    async def aks(self, data: Item, entity: str, keywords: list[str], docs: Docs):
        if self.cfg.task.method.abl.KS:
            if "Related docs" not in self.kb:
                self.kb["Related docs"] = Knowledge("Related docs")
            new_docs = [doc for doc in docs if str(doc) not in self.kb["Related docs"].contents]
            self.kb["Related docs"].contents.extend([str(doc) for doc in new_docs])
            trace = {
                "new_docs": {doc.id: str(doc) for doc in new_docs},
                "summary": "abl_learn",
            }
            return trace
        if self.cfg.task.method.abl.EI:
            # 由于 EI 的 abl, 会没有 entity 和 keywords, 所以这里摘要时，需要根据 doc.title 判断 entity
            entity2docs: dict[str, Docs] = {}
            for doc in docs:
                if doc.title not in entity2docs:
                    entity2docs[doc.title] = Docs()
                entity2docs[doc.title].add([doc])
            trace = {}
            for entity, docs in entity2docs.items():
                if entity not in self.kb:
                    self.kb[entity] = Knowledge(entity)
                new_docs = self.kb[entity].supports.add([doc for doc in docs])
                rsp = await self.agents["KS"].arun(data.id, placeholder={
                    "question": data.question,
                    "thought": "\n\n".join([f"#### step {i}\n\n{t}" for i, t in enumerate(self.thought)]),
                    "entity": entity,
                    "query": ", ".join(keywords),
                    "docs": "\n\n".join([str(doc) for doc in new_docs]),
                })
                if "None" not in rsp and "none" not in rsp:
                    self.kb[entity].contents.append(rsp)
                elif len(self.kb[entity].contents) == 0:
                    self.kb.pop(entity)
                trace[entity] = {
                    "new_docs": {idx: str(doc) for idx, doc in zip(new_docs.ids(), new_docs)},
                    "summary": rsp,
                }
            return trace
        else:
            if self.cfg.task.method.abl.KS:
                entity = "Related docs"
            if entity not in self.kb:
                self.kb[entity] = Knowledge(entity)
            new_docs = self.kb[entity].supports.add([doc for doc in docs])
            rsp = await self.agents["KS"].arun(data.id, placeholder={
                "question": data.question,
                "thought": "\n\n".join([f"#### step {i}\n\n{t}" for i, t in enumerate(self.thought)]),
                "entity": entity,
                "query": ", ".join(keywords),
                "docs": "\n\n".join([str(doc) for doc in new_docs]),
            })
            if  "None" not in rsp and "none" not in rsp:
                self.kb[entity].contents.append(rsp)
            elif len(self.kb[entity].contents) == 0:
                self.kb.pop(entity)
            trace = {
                "new_docs": {idx: str(doc) for idx, doc in zip(new_docs.ids(), new_docs)},
                "summary": rsp,
            }
            return trace


@RAGBuilder.register_module("dualrag")
class DualRAG(QA):
    def __init__(self, cfg: DictConfig, logger: Logger):
        super().__init__(cfg, logger)
        self.infer = Infer(cfg)
        self.answer = LLMAgent("answer", prompt_answer, cfg.task.base_llm)

    async def aquery(self, data: Item):
        log_dir = f"log/{data.id}/llm"
        self.logger.mkdir(log_dir)
        trace_tmp = {}
        kmanager = KManager(self.cfg)
        with tqdm(range(self.cfg.task.method.max_iter), leave=False, desc=f"{data.id}") as tbar:
            for step in tbar:
                log_step = {"knowledge": Knowledge.dict2json(kmanager.kb)}
                tbar.set_postfix(status="infer")
                thought_new, need_retrive, thought_rsp = await self.infer.ainfer(data.id, placeholder={
                    "knowledge": Knowledge.dict2str(kmanager.kb),
                    "question": data.question,
                    "thought": "\n".join(kmanager.thought),
                })
                kmanager.thought.append(thought_new)
                log_step["thought"] = thought_rsp
                if not need_retrive:
                    trace_tmp[step] = log_step
                    break
                tbar.set_postfix(status="EI")
                entity2key = await kmanager.aei(data)
                tbar.set_postfix(status="retrieve")
                log_step["retrive"] = {}
                log_step["KS"] = {}
                for entity, keywords in entity2key.items():
                    if self.cfg.task.method.abl.EI:
                        entity = " ".join(keywords)
                    log_step["retrive"][entity] = {"r": {}}
                    entity_docs = Docs()
                    for keyword in keywords:
                        docs, scores, idxs = await aretrieve(self.cfg.corpus, keyword, self.cfg.task.method.retrieve.topk)
                        entity_docs.add([Doc(idx, doc) for idx, doc in zip(idxs, docs)])
                        log_step["retrive"][entity]["r"][keyword] = {
                            "docs": {idx: doc for idx, doc in zip(idxs, docs)},
                            "scores": {idx: score for idx, score in zip(idxs, scores)},
                        }
                    if not self.cfg.task.method.retrieve.rerank_abl:
                        docs, idxs = [str(doc) for doc in entity_docs], entity_docs.ids()
                        docs, scores, idxs = await arerank(entity, idxs, docs, k=self.cfg.task.method.retrieve.rerank_topk)
                        log_step["retrive"][entity]["rerank"] = {
                            "docs": {idx: doc for idx, doc in zip(idxs, docs)},
                            "scores": {idx: score for idx, score in zip(idxs, scores)},
                        }
                    docs_to_learn = Docs([Doc(idx, doc) for idx, doc in zip(idxs, docs)])
                    if len(docs_to_learn) == 0:
                        continue
                    trace_learn = await kmanager.aks(data, entity, keywords, Docs([Doc(idx, doc) for idx, doc in zip(idxs, docs)]))
                    log_step["KS"][entity] = trace_learn
                trace_tmp[step] = log_step
        rsp = await self.answer.arun(data.id, placeholder={
            "knowledge": Knowledge.dict2str(kmanager.kb),
            "question": data.question,
            "thought": "\n\n".join([f"#### step {i}\n\n{t}" for i, t in enumerate(kmanager.thought)]),
        })
        trace = {
            "knowledge": Knowledge.dict2json(kmanager.kb),
            "thought": kmanager.thought,
            "trace": trace_tmp,
        }
        return rsp, trace
