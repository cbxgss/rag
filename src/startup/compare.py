import os
import logging
from omegaconf import DictConfig
from tqdm import tqdm
import re
import asyncio
from cr_utils import Logger

from src.startup.builder import RunBuilder
from src.evaluator.ms import MSCompare

log = logging.getLogger(__name__)


@RunBuilder.register_module("compare")
class CompareRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = Logger(cfg)
        self.logger.mkdir("output")
        os.makedirs(cfg.workspace, exist_ok=True)
        self.comparer = MSCompare(cfg)

    def run(self):
        asyncio.run(self.aprocess_all())

    async def aprocess_all(self):
        question = "Who is the mother of the director of film Polish-Russian War (Film)?"
        answer1 = "Małgorzata Braunek"
        answer2 = "The mother of the director of the film \"Polish-Russian War\" (Wojna polsko-ruska) is Małgorzata Braunek, who was an actress."
        c1, c2 = await self.comparer.cal_one(question, answer1, answer2, "1")
        print(type(c1))
        import json
        print("c1")
        print(json.dumps(c1, indent=4))
        print("c2")
        print(json.dumps(c2, indent=4))
