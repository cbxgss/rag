import os
import re
import logging
from omegaconf import DictConfig
import json
import pickle as pkl


log = logging.getLogger(__name__)


def extract_content(s: str):
    title, content = s.split("\n", 1)
    title = title.strip().strip("\"")
    new_content = f"#### {title}\n\n{content}"
    return new_content

def load_corpus(cfg: DictConfig):
    log.info(f"loading corpus {cfg.corpus}")
    pkl_path = cfg.path.corpus_pkl
    if os.path.exists(pkl_path):
        log.info("loading corpus pkl")
        with open(pkl_path, "rb") as f:
            corpus = pkl.load(f)
    else:
        os.makedirs(cfg.workspace, exist_ok=True)
        log.info("loading corpus json")
        raw_path = "download/data/RUC-NLPIR/FlashRAG_datasets/retrieval-corpus/wiki18_100w.jsonl"
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"corpus not found: {raw_path}")
        with open(raw_path, 'r') as f:
            corpus = [extract_content(json.loads(line)["contents"]) for line in f]
        with open(pkl_path, "wb") as f:
            pkl.dump(corpus, f)
            log.info("saved corpus pkl")
    return corpus
