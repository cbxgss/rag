import re
import string
from omegaconf import DictConfig
import unicodedata
from collections import Counter
import asyncio


def normalize_text(s):
    s = unicodedata.normalize('NFD', s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class BaseMetric:
    """`BaseMetric` serves as the base object of all metrics. Implemented metric should
    inherit this class.
    """

    metric_name = "base"

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dataset = cfg.dataset

    def cal_one(self, response: str, answer: str):
        raise NotImplementedError

    def cal(self, response: str, golden_answers: list[str]):
        scores = [self.cal_one(response, golden_answer) for golden_answer in golden_answers]
        return max(scores)

    async def cal_one(self, question: str, response: str, answer: str, path):
        raise NotImplementedError

    async def cal(self, question: str, response: str, golden_answers: list[str], path):
        task = [self.cal_one(question, response, golden_answer, path) for golden_answer in golden_answers]
        scores = await asyncio.gather(*task)
        return max(scores)


class F1_Score(BaseMetric):
    """Token-level F1 score"""

    metric_name = ["f1", "precision", "recall"]

    def __init__(self, cfg):
        super().__init__(cfg)

    def cal(self, response: str, golden_answers: list[str]):
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        for ground_truth in golden_answers:
            normalized_prediction = normalize_text(response)
            normalized_ground_truth = normalize_text(ground_truth)
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["f1", "precision", "recall"]:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


class ExactMatch(BaseMetric):
    r"""Exact match measure whether the predicted answer is completely consistent
    with the standard answer.

    """

    metric_name = "em"

    def __init__(self, cfg):
        super().__init__(cfg)

    def cal(self, response: str, golden_answers: list[str]) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_text(response)
        score = 0.0
        for golden_answer in golden_answers:
            golden_answer = normalize_text(golden_answer)
            if golden_answer == normalized_prediction:
                score = 1.0
                break
        return score


class Sub_ExactMatch(BaseMetric):
    r"""Sub-Exact match measure whether the predicted answer contains the standard answer."""

    metric_name = "acc"

    def __init__(self, cfg):
        super().__init__(cfg)

    def cal(self, response: str, golden_answers: list[str]) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_text(response)
        score = 0.0
        for golden_answer in golden_answers:
            golden_answer = normalize_text(golden_answer)
            if golden_answer in normalized_prediction:
                score = 1.0
                break
        return score


class Rouge_Score(BaseMetric):
    metric_name = ["rouge-1", "rouge-2", "rouge-l"]

    def __init__(self, cfg):
        super().__init__(cfg)
        from rouge import Rouge

        self.scorer = Rouge()

    def cal(self, response: str, golden_answers: list[str]):
        output = {}
        # response = normalize_text(response)
        for answer in golden_answers:
            # answer = normalize_text(answer)
            scores = self.scorer.get_scores(response, answer)
            for key in ["rouge-1", "rouge-2", "rouge-l"]:
                if key not in output:
                    output[key] = []
                output[key].append(scores[0][key]["f"])
        for k, v in output.items():
            output[k] = max(v)

        return output


class LLMJudge(BaseMetric):
    """
    Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy
    https://aclanthology.org/2023.findings-emnlp.620
    """
    metric_name = "llm_judge"
    JUDGE_PROMPT = """In the following task, you are given a Question, a model Prediction for the Question, and a Ground-truth Answer to the Question. You should decide whether the model Prediction implies the Ground-truth Answer.

Note:
For some questions, the given standard answer may not be the unique correct answer, but a possible answer. In this case, if the model's response is close in meaning to the standard answer, or contains the standard answer, then the model's response can be considered correct.

Question:
{question}

Prediction:
{response}

Ground-truth Answer
{golden_answer}

Does the Prediction imply the Ground-truth Answer? Output Yes or No:
"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.llm = cfg.task.eval_llm

    async def cal_one(self, question: str, response: str, golden_answer: str, path) -> float:
        from cr_utils import Chater
        message = [
            {"role": "user", "content": self.JUDGE_PROMPT.format(question=question, response=response, golden_answer=golden_answer)}
        ]
        while True:
            openai_params = {"temperature": 0.0}
            llm_e = await Chater().acall_llm(
                prompt=message,
                model=self.llm,
                name="evaluate",
                path=path,
                **openai_params
            )
            if "Yes" in llm_e:
                return 1.0
            elif "No" in llm_e:
                return 0.0
