from omegaconf import DictConfig
import logging
import importlib

from src.corpus.utils import extract_title, hash_object


log = logging.getLogger(__name__)


def load_corpus(cfg: DictConfig) -> list[str]:
    module_name = f"src.corpus.{cfg.corpus}"
    try:
        module = importlib.import_module(module_name)
        corpus = getattr(module, "load_corpus")(cfg)
        log.info(f"corpus length: {len(corpus)}")
        return corpus
    except Exception as e:
        log.error(f"failed to load corpus {cfg.corpus}")
        raise e
