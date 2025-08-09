import os
import logging
from omegaconf import DictConfig
from tqdm import tqdm
import json
import tarfile
import bz2
import pickle as pkl


log = logging.getLogger(__name__)


def load_corpus(cfg: DictConfig):
    """
    返回一个 list[str]，其中每个元素是一个文档
    """
    log.info(f"loading corpus {cfg.corpus}")
    pkl_path = cfg.path.corpus_pkl
    if os.path.exists(pkl_path):
        log.info("loading corpus pkl")
        with open(pkl_path, "rb") as f:
            corpus = pkl.load(f)
    else:
        os.makedirs(cfg.workspace, exist_ok=True)
        log.info("loading corpus raw")
        raw_path = "download/data/wiki/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"corpus not found: {raw_path}")
        with tarfile.open(raw_path, "r:bz2") as tar:
            corpus = []
            tar_len = len(tar.getmembers())
            log.info(f"tar len: {tar_len}")
            with tqdm(total=tar_len) as pbar:
                log.info("begin loading corpus")
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".bz2"):
                        bz2_file = tar.extractfile(member)
                        with bz2.open(bz2_file, "rt") as f:
                            for line in f:
                                js = json.loads(line)
                                text_li: list[list[str]] = js["text"]
                                text = f"#### {js['title']}\n\n"
                                for li in text_li[1:]:
                                    text += "".join(li) + "\n\n"
                                corpus.append(text)
                    pbar.update(1)
        with open(pkl_path, "wb") as f:
            pkl.dump(corpus, f)
            log.info("saved corpus pkl")
    return corpus
