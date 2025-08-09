from omegaconf import DictConfig
import logging
from tqdm.asyncio import tqdm as atqdm
from cr_utils import Logger

from src.dataset import Item
from src.rag.base import QA
from src.rag.metarag.utils import format_ref, check_answer
from src.rag.metarag.llms import LLM
from src.rag.metarag.WikiSearcher import WikiSearcher
from src.rag.metarag.monitor import monitor
from src.rag.metarag.generator import generator
from src.rag.metarag.critic import critic
from src.startup import RAGBuilder

from src.tools import anli


log = logging.getLogger(__name__)


@RAGBuilder.register_module("metarag")
class MetaRAG(QA):
    def __init__(
            self,
            cfg: DictConfig,
            logger: Logger,
    ):
        super().__init__(cfg, logger)
        self.llm_name = self.cfg.task.base_llm
        self.max_iter = self.cfg.task.method.max_iter
        self.topk = self.cfg.task.retrieve.topk
        self.ref_num = self.cfg.task.retrieve.rerank_topk
        self.threshold = self.cfg.task.method.threshold
        self.expert_model = self.cfg.task.method.expert_model

        # load models
        print("Loading LLM...")
        self.llm = LLM(self.llm_name)

        self.retriever = WikiSearcher(cfg)
        print("Loading monitor...")
        self.monitor = monitor(expert_model_name = self.expert_model, device = self.cfg.task.method.device.monitor)
        print("Loading generator...")
        self.generator = generator(llm = self.llm)
        print("Loading critic...")
        self.critic = critic(llm = self.llm)

    async def aadd_new_reference(self, id, question:str, reference:list ,single_log:dict, answer:str):
        rewrite_query = await self.critic.arewrite(id, question, answer, reference)
        rewrite_query = rewrite_query[:64]
        new_reference, trace = await self.retriever.asearch(rewrite_query,self.topk, self.ref_num)

        single_log['add_reference'] = new_reference

        reference = reference + new_reference
        reference = list(set(reference))

        return reference, single_log


    async def apredict(self, data:Item):
        question = data.question
        logs = []
        output_item = {'question': question}
        tbar = atqdm(total=self.max_iter, desc=f"Processing {data.id}", leave=False)
        tbar.set_postfix(status="init_retrieve")
        reference, trace = await self.retriever.asearch(question, 100, self.ref_num)
        output_item['reference'] = reference
        tbar.set_postfix(status="init_answer")
        reason, answer = await self.generator.aanswer(data.id, question, format_ref(reference), suggestion=None, hint=None)
        final_answer = answer

        while True:
            tbar.update(1)
            single_log = {"original_output": {"reason":reason,"answer":answer}}

            tbar.set_postfix(status="monitor")
            # monitor_result: {"pseduo_answer":...,"score":score from 0 to 1}
            # 计算与专家模型输出的 sim score
            monitor_result = self.monitor.judge(question, format_ref(reference), answer)
            single_log['monitor_result'] = monitor_result
            monitor_judge = True if monitor_result['score'] > self.threshold else False
            if monitor_judge or len(logs) >= self.max_iter:
                final_answer = answer
                logs.append(single_log)
                break

            tbar.set_postfix(status="critic")
            # critic_result: {"internal_judgement":..., "external_judgement":..., "judgement":...,"feedback":/.}
            """
            1. 先检查基本错误: Logic error, 不符合问题要求，回答冗余等, 依赖于 few-shot
            2. 用 llm，判断 I/E 知识是否足够
            """
            critic_result = await self.critic.afeedback(data.id, question, format_ref(reference), answer)
            if critic_result['judgement'] == "correct":
                hint = "Please think step by step."
            else:
                hint = critic_result['feedback']

            if critic_result['internal_judge']:
                if critic_result['external_judge']: #  I E 都全，利用 NLI 模型检索 answer
                    suggestion = "Carefully check your final answer. Do not add irrelevant content and answer questions accurately."
                    reason_support = await self.judge_support(reason,answer)
                    critic_result['reason_support'] = reason_support
                    if not reason_support:
                        suggestion = "Please try to provide an answer by considering multi-step reasoning."
                else:       # I 全 E 不全，允许 llm 利用自己的知识, 同时继续拆分子问题，继续检索
                    suggestion = "If you feel that there is no suitable content in the reference, try relying on yourself to answer the question."                    # 添加新的reference
                    reference, single_log = await self.aadd_new_reference(data.id, question, reference, single_log, answer)
                    output_item['reference'] = reference
            else:
                if critic_result['external_judge']: # I 不全 E 全，完全依赖 reference
                    suggestion = "You need to answer questions entirely based on reference, and do not use your own knowledge."
                else:    # I E 都不全，拆分子问题，继续检索
                    suggestion = "You can break down the question into sub questions to use reference."
                    reference, single_log = await self.aadd_new_reference(data.id, question, reference, single_log, answer)
                    output_item['reference'] = reference

            tbar.set_postfix(status="answer")
            new_reason, new_answer = await self.generator.aanswer(data.id, question, format_ref(reference), suggestion, hint)
            if check_answer(new_answer):
                final_answer = new_answer
            single_log['critic_result'] = critic_result
            single_log['new_answer'] = {"reason":new_reason,"answer":new_answer}

            logs.append(single_log)
            reason,answer = new_reason,new_answer

        output_item['final_answer'] = final_answer
        output_item['interaction_log'] = logs

        return output_item

    async def judge_support(self, reason: str, answer: str) -> bool:
        inference = await anli(premise=reason, hypothesis=answer)
        return inference

    async def aquery(self, data: Item):

        output_item = await self.apredict(data)
        rsp = output_item['final_answer']
        trace = {k: v for k, v in output_item.items() if k != 'final_answer'}
        return rsp, trace
