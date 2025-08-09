import os
import logging
from cr_utils import Chater


log = logging.getLogger(__name__)


class LLMAgent:
    def __init__(self, name: str, prompt: str, llm: str, response_format: dict = None, _log=True):
        self.name = name
        self.prompt = prompt
        self.llm = llm
        self.response_format = response_format
        self.log = _log

    async def arun(self, id, placeholder: dict, openai_params: dict = None):
        openai_params = openai_params or {}
        messages = [
            {"role": "user", "content": self.prompt.format_map(placeholder)},
        ]
        if self.response_format:
            openai_params["response_format"] = self.response_format
        rsp = await Chater().acall_llm(
            prompt=messages,
            model=self.llm,
            name=self.name,
            path=f"log/{id}/llm",
            **openai_params,
        )
        return rsp

def prompt(path: str, en: bool = False) -> str:
    if en:
        path = path.replace("prompt", "prompt_en")
    with open(path, 'r') as file:
        return file.read()


async def atranslate(path: str) -> str:
    path_en = path.replace("prompt", "prompt_en")
    if not os.path.exists(path_en):
        os.makedirs(os.path.dirname(path_en), exist_ok=True)
        translate_prompt = """把下面的 prompt 内容翻译成英文，格式不要做任何改动。

{content}
"""
        translator = LLMAgent("translator", translate_prompt, "gpt-4o")
        prompt_zh = prompt(path)
        prompt_en = await translator.arun("trans", content=prompt_zh)
        with open(path_en, 'w') as file:
            file.write(prompt_en)
