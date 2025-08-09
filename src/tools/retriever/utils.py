import os
from omegaconf import DictConfig
import logging
import faiss

from src.tools.embedder import FastApiEmbedder, load_embedding


log = logging.getLogger(__name__)


def build_index(cfg: DictConfig):
    index_path = cfg.path.index
    if os.path.exists(index_path):
        log.info(f"Index already exists at {index_path}")
    else:
        embedding_path = cfg.path.embedding
        log.info(f"Loading embeddings from {embedding_path}")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        embedder = FastApiEmbedder()
        embs = load_embedding(embedding_path, embedder.get_dim())

        log.info(f"Creating index at {index_path}")
        dim = embs.shape[-1]
        faiss_index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
        faiss_gpu = None
        if faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(embs)
            faiss_index.add(embs)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(embs)
            faiss_index.add(embs)
        faiss.write_index(faiss_index, index_path)
        log.info("Finish!")
