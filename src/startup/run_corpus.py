import logging
from omegaconf import DictConfig

from src.startup.builder import RunBuilder
from src.corpus import load_corpus
from src.tools.embedder import build_embedding
from src.tools.retriever.utils import build_index


log = logging.getLogger(__name__)


@RunBuilder.register_module("corpus")
class CorpusRuner:
    def __init__(self, cfg: DictConfig):
        log.info(f"Initializing {self.__class__.__name__} {cfg.corpus}")
        self.cfg = cfg

    def run(self):
        log.info("building corpus")
        corpus = load_corpus(self.cfg)
        log.info(f"corpus len: {len(corpus)}")
        build_embedding(self.cfg, corpus, batch_size=self.cfg.task.batch_size)
        log.info("building index")
        build_index(self.cfg)
