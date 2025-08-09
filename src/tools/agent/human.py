import logging
import json
from omegaconf import DictConfig
from cr_utils import Logger


log = logging.getLogger(__name__)


class Human:
    def __init__(self, cfg: DictConfig, logger: Logger, name: str, prompt: str, out_json: bool = False):
        self.cfg = cfg
        self.logger = logger
        self.name = name
        self.prompt = prompt
        self.out_json = out_json

    def run(self):
        log.info(f"Human {self.name}: {self.prompt}")
        return self._input() if not self.out_json else self._input_json()

    @staticmethod
    def _input():
        rsp = input("input str: ")
        return rsp

    @staticmethod
    def _input_json():
        while True:
            rsp = input("input json: ")
            try:
                rsp = json.loads(rsp)
                return rsp
            except Exception as e:
                log.error(f"json error: {e} when decode {rsp}")
