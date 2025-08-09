import json
from tenacity import retry, stop_after_delay, wait_random_exponential

from src.rag.metarag.config import QA_PROMPT, QA_response_format


class generator:
    def __init__(self, llm):
        self.llm = llm

    @retry(stop=stop_after_delay(10), wait=wait_random_exponential(multiplier=1, min=1, max=10))
    async def aanswer(self, id, question, reference, suggestion=None, hint=None):
        suggestion_text = f"Here are some suggestions you need to follow: {suggestion}\n" if suggestion else ""
        hint_text = f"Here are some mistakes you may make, you need to be careful: {hint}\n" if hint else ""
        all_hint = suggestion_text + hint_text
        qa_prompt = QA_PROMPT.format(reference=reference, question=question, all_hint=all_hint)

        output = await self.llm.aget_output(
            id, "generator", qa_prompt, system_prompt="Act as an LLM helper.",
            response_format=QA_response_format
        )
        js = json.loads(output)
        return js['reason'], js['answer']
