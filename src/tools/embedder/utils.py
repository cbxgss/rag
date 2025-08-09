import os
import logging
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np

from src.tools.embedder import FastApiEmbedder


log = logging.getLogger(__name__)


def build_embedding(cfg: DictConfig, corpus: list[str], batch_size: int):
    embedding_path = cfg.path.embedding
    if os.path.exists(embedding_path):
        log.info(f"embedding already exists, skip")
    else:
        log.info(f"creating embedding: {embedding_path}")
        embedder = FastApiEmbedder()
        total_size, embedding_dim = len(corpus), embedder.get_dim()
        memmap = np.memmap(
            embedding_path,
            shape=(total_size, embedding_dim),
            mode="w+",
            dtype=np.float32
        )
        for i in tqdm(range(0, total_size, batch_size), desc=f"corpus embedding in batch_size={batch_size}"):
            batch_corpus = corpus[i:i + batch_size]
            batch_embs = embedder.create_embedding(batch_corpus, batch_size=batch_size)
            start_idx = i
            end_idx = min(i + batch_size, total_size)
            if batch_embs.shape != (end_idx - start_idx, embedding_dim):
                raise ValueError(f"batch_embs.shape {batch_embs.shape} != {(end_idx - start_idx, embedding_dim)}")
            memmap[start_idx:end_idx] = batch_embs
            memmap.flush()


def load_embedding(embedding_path, embedding_dim) -> np.ndarray:
    all_embeddings = np.memmap(
        embedding_path,
        mode="r",
        dtype=np.float32
    ).reshape(-1, embedding_dim)
    log.info(f"emb.shape: {all_embeddings.shape}")
    return all_embeddings
