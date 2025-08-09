import logging
from omegaconf import DictConfig
from tqdm import tqdm
import re
import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
import tiktoken
from litellm import Choices
from cr_utils import Logger, Chater

from src.dataset import Item
from src.tools import aretrieve, arerank
from src.rag.base import QA
from src.startup import RAGBuilder


log = logging.getLogger(__name__)


class Generator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        model_name = cfg.task.base_llm
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            model_name_raw ={
                "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
                "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
                "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
                "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
            }[model_name]
            log.info(f"Failed to load tokenizer for model {model_name}, trying AutoTokenizer {model_name_raw}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_raw)

    async def agenerate(self, id, cost_name: str, message: list, return_scores=False, stop=None, max_new_tokens=None):
        openai_params = {"max_tokens": max_new_tokens, "logprobs": return_scores, "stop": stop}
        choice: Choices = await Chater().acall_llm(message, model=self.cfg.task.base_llm, name=cost_name, path=f"log/{id}/llm", **openai_params, return_all=True)
        content, logprobs = choice.message.content, np.exp(list(map(lambda x: x.logprob, choice.logprobs.content)))
        return content, logprobs


def get_message(question: str, retrieval_result: list[str], previous_gen: str):
    system_prompt = """You are a RAG chatbot that will answer questions based on the relevant documents provided.

Relevant documents:

{reference}
"""
    base_prompt = """Question: {question}

Your answer:

{previous_gen}

Please continue the answer based on the given documents.
"""
    message = [
        {"role": "system", "content": system_prompt.format(reference="\n\n".join(retrieval_result))},
        {"role": "user", "content": base_prompt.format(question=question, previous_gen=previous_gen)}
    ]
    return message


class Retriever:
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


@RAGBuilder.register_module("flare")
class FLARE(QA):
    def __init__(self, cfg: DictConfig, logger: Logger):
        super().__init__(cfg, logger)
        self.method_config: DictConfig = cfg.task.method

        self.generator = Generator(cfg)
        self.retriever = Retriever(cfg)

        self.threshold = self.method_config.threshold
        self.max_generation_length = self.method_config.max_generation_length
        self.look_ahead_steps = self.method_config.look_ahead_steps
        self.max_iter_num = self.method_config.max_iter_num
        self.stop_sym = list("!@#$%^&*()\n\n)(*&^%$#@!")
        self.stop_sym = list("!\n")

        self.topk = cfg.task.retrieve.topk
        self.rerank_topk = cfg.task.retrieve.rerank_topk

    def get_next_sentence(self, output, scores):
        tokenizer = self.generator.tokenizer
        text_sentences = re.split(r"(?<=[^A-Z].[.?]) +", output)
        if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            token_id_sentences = [tokenizer.encode(s, add_special_tokens=False) for s in text_sentences]
        else:
            token_id_sentences = [tokenizer.encode(s, allowed_special="all") for s in text_sentences]

        first_sent_ids = token_id_sentences[0]
        first_sent_score = scores[: len(first_sent_ids)]

        return text_sentences[0], first_sent_score

    def judge_sent_confidence(self, sent, sent_score):
        judge_result = all([score > self.threshold for score in sent_score])
        new_query = None
        if not judge_result:
            tokenizer = self.generator.tokenizer
            if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                sent_ids = tokenizer.encode(sent, add_special_tokens=False)
            else:
                sent_ids = tokenizer.encode(sent, allowed_special="all")
            # assert len(sent_ids) == len(sent_score)
            new_query_ids = [i for i, score in zip(sent_ids, sent_score) if score > self.threshold]
            new_query = tokenizer.decode(new_query_ids)
            if len(new_query) == 0:
                judge_result = True
        return judge_result, new_query

    async def aquery(self, data: Item):
        question = data.question
        gen_length = 0
        iter_round = 0
        final_gen_result = ""
        trace = {}
        tbar = tqdm(range(self.max_iter_num), leave=False, desc=f"{data.id}")
        while gen_length < self.max_generation_length and iter_round < self.max_iter_num:
            tbar.update(1)
            trace_step = {}
            message = get_message(question=question, retrieval_result=[], previous_gen=final_gen_result)
            round_gen_output, scores = await self.generator.agenerate(
                data.id, cost_name="generate",
                message=message, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps,
            )
            trace_step["gen"] = round_gen_output
            # next_sent_scores: token logits of the first sent in generation seq
            next_sent, next_sent_score = self.get_next_sentence(round_gen_output, scores)
            trace_step["next_sent"] = next_sent
            trace_step["next_sent_score"] = next_sent_score
            # judge next sentence
            judge_result, query = self.judge_sent_confidence(next_sent, next_sent_score)
            trace_step["judge_result"] = judge_result

            if not judge_result:
                # do retrieval-augmented generation
                retrieval_result, trace_r = await self.retriever.asearch(query, self.topk, self.rerank_topk)
                trace_step["retrieval"] = trace_r
                message = get_message(question=question, retrieval_result=retrieval_result, previous_gen=final_gen_result)
                output, scores = await self.generator.agenerate(
                    data.id, cost_name="generate",
                    message=message, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps
                )
                next_sent, _ = self.get_next_sentence(output, scores)

            final_gen_result += next_sent
            gen_length += len(next_sent_score)
            iter_round += 1

        return final_gen_result, trace
