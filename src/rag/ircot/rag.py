from omegaconf import DictConfig
from cr_utils import Logger

from src.dataset import Item
from src.tools import LLMAgent, aretrieve, arerank
from src.rag.base import QA
from src.rag.ircot.struct import Doc, Docs
from src.startup import RAGBuilder


@RAGBuilder.register_module("ircot")
class IRCOT(QA):
    IRCOT_INSTRUCTION = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
    IRCOT_EXAMPLE = "Wikipedia Title: Kurram Garhi\nKurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\nWikipedia Title: 2001–02 UEFA Champions League second group stage\nEight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.\n\nWikipedia Title: Satellite tournament\nA satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.\n\nWikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\nWikipedia Title: Telephone numbers in Ascension Island\nCountry Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.\n\nQuestion: Are both Kurram Garhi and Trojkrsti located in the same country?\nThought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.\n\n"

    def __init__(self, cfg: DictConfig, logger: Logger):
        super().__init__(cfg, logger)
        self.prompt = f"{self.IRCOT_INSTRUCTION}\n\n{self.IRCOT_EXAMPLE}\n\n"
        self.prompt += "### Background Knowledge\n\n{reference}\n\n### Question: {question}\nThought:{thoughts}"
        self.agent = LLMAgent("ircot", self.prompt, cfg.task.base_llm)

    async def aquery(self, data: Item):
        log_dir = f"log/{data.id}/llm"
        self.logger.mkdir(log_dir)
        trace = {}
        docs, scores, idxs, trace_step = await self.aretrieve(data.question)
        docs_all = Docs([Doc(idx, doc) for idx, doc in zip(idxs, docs)])
        trace[1] = {
            "thought": "",
            "retrieve": trace_step
        }
        thoughts = []
        iter_num = 1
        while iter_num <= self.cfg.task.method.max_iter:
            placeholder = {
                "question": data.question,
                "reference": str(docs_all),
                "thoughts": " ".join(thoughts),
            }
            rsp = await self.agent.arun(data.id, placeholder, openai_params={"stop": [".", "\n"], "temperature": 0.3}) + "."
            thoughts.append(rsp)
            iter_num += 1
            if "So the answer is:" in rsp:
                break
            docs, scores, idxs, trace_step = await self.aretrieve(rsp)
            docs_all = Docs([Doc(idx, doc) for idx, doc in zip(idxs, docs)])
            trace[iter_num] = {
                "thought": rsp,
                "retrieve": trace_step
            }
        return " ".join(thoughts), trace

    async def aretrieve(self, query: str):
        trace = {}
        docs, scores, idxs = await aretrieve(self.cfg.corpus, query, self.cfg.task.method.retrieve.topk)
        trace["retrieve"] = {
            "docs": {idx: doc for idx, doc in zip(idxs, docs)},
            "scores": {idx: score for idx, score in zip(idxs, scores)},
        }
        docs, scores, idxs = await arerank(query, idxs, docs, k=self.cfg.task.method.retrieve.rerank_topk, filter=self.cfg.task.method.retrieve.rerank_filter)
        trace["rerank"] = {
            "docs": {idx: doc for idx, doc in zip(idxs, docs)},
            "scores": {idx: score for idx, score in zip(idxs, scores)},
        }
        return docs, scores, idxs, trace
