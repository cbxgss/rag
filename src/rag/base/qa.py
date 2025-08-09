direct_system = """You are a chatbot that will answer questions.
Please output the answer in a few words, keeping it as brief as possible.
For some questions, simply respond with "yes" or "no."
"""

direct_user = """question:
{query}
"""


from omegaconf import DictConfig
import logging
from cr_utils import Logger, Chater

from src.dataset import Item
from src.startup import RAGBuilder


log = logging.getLogger(__name__)


@RAGBuilder.register_module("direct")
class QA:
    def __init__(self, cfg: DictConfig, logger: Logger):
        self.cfg = cfg
        self.logger = logger

    async def aquery(self, data: Item):
        log_dir = f"log/{data.id}/llm"
        self.logger.mkdir(log_dir)
        messages = [
            {"role": "system", "content": direct_system},
            {"role": "user", "content": direct_user.format(query=data.question)},
        ]
        response = await Chater().acall_llm(
            prompt=messages,
            model=self.cfg.task.base_llm,
            name="generate",
            path=log_dir,
            temperature=0.0,
            max_tokens=1024
        )
        return response, {}
