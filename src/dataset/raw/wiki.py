import os
import logging
from omegaconf import DictConfig
import json
import pandas as pd


log = logging.getLogger(__name__)


def trans_corpus(cfg: DictConfig):
    if os.path.exists(cfg.dataset.corpus.path):
        log.info("corpus already exists, skip")
    else:
        log.info("loading raw data")
        df = pd.read_csv(cfg.dataset.corpus.path_raw, sep='\t')
        corpus_json = {row["id"]: {"title": row["title"], "text": row["text"]} for _, row in df.iterrows()}
        with open(cfg.dataset.corpus.path, 'w') as f:
            json.dump(corpus_json, f, indent=4, ensure_ascii=False)
