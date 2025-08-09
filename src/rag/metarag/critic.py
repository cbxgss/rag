import re
import json

from src.rag.metarag.config import (
    CHECK_PROMPT, CHECK_response_format,
    REWRITE_PROMPT, INTERNAL_PROMPT, EXTERNAL_PROMPT
)


def parse_llm_output_rewrite(output):
    pattern = r"(.*?The rewrite query is.*?)"
    # 使用正则表达式进行匹配
    match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
    if match:
        query = match.group(1).strip()
    else:
        query = output.replace("the rewrite query is","")
    return query


class critic:
    def __init__(self, llm):
        self.llm = llm

    async def acheck(self, id, question, answer):
        check_prompt = CHECK_PROMPT.format(question = question, answer=answer)
        judge = await self.llm.aget_output(
            id, "check", check_prompt, system_prompt="Act as an Evaluator & Critic System.",
            response_format=CHECK_response_format
        )
        judge = json.loads(judge)
        return judge

    async def ajudge_internal(self, id, question):
        prompt = INTERNAL_PROMPT.format(question = question)

        judge = await self.llm.aget_output(id, "judge_i", prompt, system_prompt="Act as an Evaluator & Critic System.")
        check_list = ["sorry","couldn't","don't have access","no"]
        for keyword in check_list:
            if keyword in judge:
                return False
        return True


    async def ajudge_external(self, id, question, reference):
        prompt = EXTERNAL_PROMPT.format(question = question, reference = reference)
        output = (await self.llm.aget_output(id, "judge_e", prompt, system_prompt="Act as an Evaluator & Critic System.")).strip().lower()
        if "no" in output:
            return False
        else:
            return True

    async def afeedback(self, id, question, reference, answer):
        basic_judge = await self.acheck(id, question, answer)

        internal_judge = await self.ajudge_internal(id, question)
        external_judge = await self.ajudge_external(id, question, reference)

        basic_judge['internal_judge'] = internal_judge
        basic_judge['external_judge'] = external_judge

        return basic_judge

    async def arewrite(self, id, question, answer, reference):
        rewrite_prompt = REWRITE_PROMPT.format(question = question, answer = answer, reference = "\n".join(reference))
        new_query = parse_llm_output_rewrite(await self.llm.aget_output(id, "rewrite", rewrite_prompt))
        return new_query
