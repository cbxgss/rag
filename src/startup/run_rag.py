import os
import logging
from omegaconf import DictConfig
from tqdm.asyncio import tqdm
from copy import deepcopy
import re
import pandas as pd
from prettytable import PrettyTable
import asyncio
from cr_utils import Logger, CostManagers

from src.startup.builder import RunBuilder, RAGBuilder
from src.rag import QA
from src.tools import LLMAgent
from src.dataset import get_dataset, Item
from src.evaluator import BaseMetric, ExactMatch, Sub_ExactMatch, F1_Score, Rouge_Score, LLMJudge


log = logging.getLogger(__name__)


@RunBuilder.register_module("rag")
class RagRunner:
    def __init__(self, cfg: DictConfig):
        log.info("Initializing Runner")
        self.cfg = cfg
        self.logger = Logger(cfg)
        self.lock = asyncio.Lock()
        self.logger.mkdir("output")
        os.makedirs(cfg.workspace, exist_ok=True)
        log.info(f"dataset: {cfg.dataset}")
        log.info(f"all methods: {RAGBuilder.module_dict.keys()}")
        log.info(f"method: {cfg.task.method.name}")
        rag_cls = RAGBuilder.get(cfg.task.method.name)
        self.rag: QA = rag_cls(cfg, self.logger)
        self.qa, extract_answer_prompt = get_dataset(cfg)
        self.reanswer = LLMAgent("reanswer", extract_answer_prompt, cfg.task.base_llm)
        self.evaluators: dict[str, BaseMetric] = {
            "LLMJudge": LLMJudge(cfg),
            "ExactMatch": ExactMatch(cfg),
            "Sub_ExactMatch": Sub_ExactMatch(cfg),
            "F1_Score": F1_Score(cfg),
            # "Rouge_Score": Rouge_Score(cfg),
        }
        self.metrics = []
        for evaluator in self.evaluators.values():
            self.metrics += evaluator.metric_name if isinstance(evaluator.metric_name, list) else [evaluator.metric_name]
        self.evaluate_df = pd.DataFrame(columns=["id"] + self.metrics)

    def run(self):
        asyncio.run(self.aprocess_all())

    async def aprocess_all(self):
        bs = self.cfg.task.batch_size
        exp_size = self.cfg.task.exp_size if self.cfg.task.exp_size > 0 else len(self.qa)
        with tqdm(range(0, len(self.qa), bs), desc=f"batch_size: {bs}") as tbar:
            for data_i in tbar:
                if data_i > exp_size: continue
                batch_l, batch_r = data_i, min(data_i + bs, len(self.qa), exp_size)
                tbar.set_postfix_str(f"batch: [{batch_l}, {batch_r})")
                data_batch = self.qa[batch_l:batch_r]
                await self.arun_batch(data_batch)

    async def arun_one(self, data: Item):
        response, trace = await self.rag.aquery(data)
        flag_re = True
        if self.cfg.task.method.name in ["direct", "native", "oracle"]:
            flag_re = False
        if self.cfg.dataset in ["eli5", "asqa"]:
            flag_re = False
        response_re = response if not flag_re else await self.reanswer.arun(data.id, placeholder={
            "question": data.question,
            "response": response,
        })
        evaluate = await self.aevaluate(data.question, response, data.golden_answers, f"log/{data.id}/llm")
        if flag_re:
            evaluate_re = await self.aevaluate(data.question, response_re, data.golden_answers, f"log/{data.id}/llm")
            evaluate = [max(evaluate[i], evaluate_re[i]) for i in range(len(evaluate))]
        out = {
            "id": data.id,
            "question": data.question,
            "response_re": response_re,
            "response": response,
            "evaluate": evaluate,
            "golden_answers": data.golden_answers,
            "trace": trace,
            "metadata": data.metadata,
        }
        self.logger.save_json(f"output/{data.id}.json", out)
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        async with self.lock:
            self.evaluate_df.loc[len(self.evaluate_df)] = [str(data.id), *evaluate]
            self.evaluate_df = self.evaluate_df.sort_values(by="id", key=lambda col: col.map(natural_sort_key))
            tmp = deepcopy(self.evaluate_df)
            tmp.loc[len(self.evaluate_df)] = self.evaluate_mean()
            self.logger.save_csv("evaluate.csv", tmp)
        return [data.id, *evaluate]

    async def arun_batch(self, data_batch: list[Item]):
        task = []
        for data in data_batch:
            task.append(self.arun_one(data))
        evaluate_batch = await asyncio.gather(*task)
        table = PrettyTable()
        table.field_names = ["id"] + self.metrics
        for row in evaluate_batch:
            table.add_row(row)
        table.add_row(self.evaluate_mean())
        log.info(f"evaluate table:\n{table}")
        cost_manager = CostManagers()
        cost_manager.show_cost()

    async def aevaluate(self, question: str, response: str, golden_answers: list[str], path) -> list[float]:
        local = {}
        for evaluator in self.evaluators.values():
            if isinstance(evaluator, LLMJudge):
                local[evaluator.metric_name] = await evaluator.cal(question, response, golden_answers, path)
            else:
                ret = evaluator.cal(response, golden_answers)
                if isinstance(evaluator.metric_name, list):
                    for metric in evaluator.metric_name:
                        local[metric] = ret[metric]
                else:
                    local[evaluator.metric_name] = ret
        evaluate = [local[metric] for metric in self.metrics]
        return evaluate

    def evaluate_mean(self):
        return ['mean'] + self.evaluate_df[self.metrics].mean().tolist()
