import importlib
import os
import logging
import json
from omegaconf import DictConfig
import pickle as pkl

from src.dataset.dataset import Dataset


log = logging.getLogger(__name__)


def load_corpus(cfg: DictConfig):
    log.info(f"loading corpus {cfg.dataset.corpus.name}")

    pkl_path = os.path.join(cfg.corpus_dir, "corpus.pkl")
    if os.path.exists(pkl_path):
        log.info("loading corpus pkl")
        with open(pkl_path, "rb") as f:
            corpus = pkl.load(f)
    else:
        log.info("loading corpus json")
        with open(cfg.dataset.corpus.path, 'r') as f:
            datasets = json.load(f)
            corpus = [v["text"] for k, v in datasets.items()]
        os.makedirs(cfg.corpus_dir, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pkl.dump(corpus, f)
            log.info("saved corpus pkl")
    log.info(f"corpus length: {len(corpus)}")
    return corpus


def get_dataset(cfg: DictConfig) -> tuple[Dataset, str]:
    module_name = f"src.dataset.{cfg.dataset}"
    try:
        module = importlib.import_module(module_name)
        dataset, reanswer = getattr(module, f"get_{cfg.dataset}")()
        return dataset, reanswer
    except Exception as e:
        log.error(f"failed to load dataset {cfg.dataset}")
        raise e
