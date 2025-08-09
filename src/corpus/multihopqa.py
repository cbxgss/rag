import os
import logging
from tqdm import tqdm
from omegaconf import DictConfig
import json
import pickle as pkl
from chonkie import WordChunker
from tokenizers import Tokenizer


from src.corpus.utils import hash_object


log = logging.getLogger(__name__)


def load_corpus(cfg: DictConfig):
    log.info(f"loading corpus {cfg.corpus}")
    pkl_path = cfg.path.corpus_pkl
    if os.path.exists(pkl_path):
        log.info("loading corpus pkl")
        with open(pkl_path, "rb") as f:
            corpus = pkl.load(f)
    else:
        os.makedirs(cfg.workspace, exist_ok=True)
        log.info("loading corpus raw")
        base_path = os.path.join(
            os.getenv("HF_HOME"), "hub", "datasets--yixuantt--MultiHopRAG", "snapshots", "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82",
            "corpus.json"
        )
        if not os.path.exists(base_path):
            raise Exception("not exist multihotpq")
        # 分词 chunk
        tokenizer = Tokenizer.from_pretrained("gpt2")
        chunker = WordChunker(tokenizer, chunk_size=250, chunk_overlap=50)
        # 构建 corpus
        used_full_ids, corpus = set(), []
        with open(base_path, "r") as f:
            corpurs_raw = json.load(f)
            tbar = tqdm(corpurs_raw)
            for item in tbar:
                title, content = item["title"], item["body"]
                chunks = chunker.chunk(content)
                for chunk in chunks:
                    chunk_str = chunk.text
                    full_id = hash_object(" ".join([title, chunk_str]))
                    if full_id in used_full_ids:
                        continue
                    used_full_ids.add(full_id)
                    doc = f"""#### {title}
- Author: {item['author']}
- Source: {item['source']}
- Published_Date: {item['published_at']}
- Content: {chunk_str}"""
                    corpus.append(doc)
                    tbar.set_postfix(len=len(corpus))
        with open(pkl_path, "wb") as f:
            pkl.dump(corpus, f)
            log.info("saved corpus pkl")
    log.info(f"corpus length: {len(corpus)}")
    return corpus
