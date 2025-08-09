from tenacity import retry, stop_never, wait_random_exponential
from cr_utils import Chater


class LLM:
    def __init__(self, llm_name:str):
        self.model_name = llm_name

    @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
    async def aget_output(self, id, cost_name: str, prompt, system_prompt="Act as an Evaluator & Critic System.", response_format: dict=None):
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        args = {"response_format": response_format} if response_format is not None else {}
        openai_params = {"temperature": 0.0, "max_tokens": 1024, **args}
        rsp = await Chater().acall_llm(message, model=self.model_name, name=cost_name, path=f"log/{id}/llm", **openai_params)
        return rsp
