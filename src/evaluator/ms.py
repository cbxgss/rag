from omegaconf import DictConfig
from cr_utils import Chater

class MSCompare:
    """
    LightRAG: Simple and Fast Retrieval-Augmented Generation
    """
    metric_name = "ms_compare"
    prompt = """### Role

You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

### Task

You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{question}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.
"""
# Output your evaluation in the following JSON format:

# {{
#     "Comprehensiveness": {{
#         "Winner": "[Answer 1 or Answer 2]",
#         "Explanation": "[Provide explanation here]"
#     }},
#     "Empowerment": {{
#         "Winner": "[Answer 1 or Answer 2]",
#         "Explanation": "[Provide explanation here]"
#     }},
#     "Overall Winner": {{
#         "Winner": "[Answer 1 or Answer 2]",
#         "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
#     }}
# }}
    response_format= {
        "type": "json_schema",
        "json_schema": {
            "name": "compare",
            "schema": {
                "type": "object",
                "properties": {
                    "Comprehensiveness": {
                        "type": "object",
                        "properties": {
                            "Winner": {"type": "string"},
                            "Explanation": {"type": "string"}
                        },
                        "required": ["Winner", "Explanation"],
                        "additionalProperties": False
                    },
                    "Empowerment": {
                        "type": "object",
                        "properties": {
                            "Winner": {"type": "string"},
                            "Explanation": {"type": "string"}
                        },
                        "required": ["Winner", "Explanation"],
                        "additionalProperties": False
                    },
                    "Overall Winner": {
                        "type": "object",
                        "properties": {
                            "Winner": {"type": "string"},
                            "Explanation": {"type": "string"}
                        },
                        "required": ["Winner", "Explanation"],
                        "additionalProperties": False
                    }
                },
                "required": ["Comprehensiveness", "Empowerment", "Overall Winner"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    def __init__(self, cfg: DictConfig):
        self.llm = cfg.task.base_llm

    async def cal_one(self, question: str, answer1: str, answer2: str, path) -> int:
        message1 = [
            {"role": "user", "content": self.prompt.format(question=question, answer1=answer1, answer2=answer2)}
        ]
        message2 = [
            {"role": "user", "content": self.prompt.format(question=question, answer1=answer2, answer2=answer1)}
        ]
        c1 = await Chater().acall_llm(prompt=message1, model=self.llm, name="ms_compare", path=path, response_format=self.response_format, temperature=0.0)
        c2 = await Chater().acall_llm(prompt=message2, model=self.llm, name="ms_compare", path=path, response_format=self.response_format, temperature=0.0)
        c2 = c2.replace("Answer 1", "Answer 3").replace("Answer 2", "Answer 1").replace("Answer 3", "Answer 2")
        c1, c2 = eval(c1), eval(c2)
        return c1, c2
