import os
import logging
from tqdm import tqdm
from omegaconf import DictConfig
import json
import pickle as pkl

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
            os.getenv("HF_HOME"), "hub", "datasets--cbxgss--rag", "snapshots", "64d4a872814da55c8284f5536795df03c39ddad2",
            "musique"
        )
        if not os.path.exists(base_path):
            raise Exception("not exist musique")
        raw_filepaths = [
            os.path.join(base_path, "train.jsonl"),
            os.path.join(base_path, "dev.jsonl"),
        ]
        used_full_ids, corpus = set(), []
        tbar1 = tqdm(raw_filepaths)
        for raw_filepath in tbar1:
            tbar1.set_postfix(file=raw_filepath)
            with open(raw_filepath, "r") as f:
                data = [json.loads(line) for line in f]
                tbar2 = tqdm(data, leave=False)
                for item in tbar2:
                    question_decomposition = item["metadata"]["question_decomposition"]
                    for sub_q in question_decomposition:
                        title = sub_q["support_paragraph"]["title"]
                        paragraph_text = sub_q["support_paragraph"]["paragraph_text"]
                        full_id = hash_object(" ".join([title, paragraph_text]))
                        if full_id in used_full_ids:
                            continue
                        used_full_ids.add(full_id)
                        doc = f"#### {title}\n\n{paragraph_text}"
                        corpus.append(doc)
        with open(pkl_path, "wb") as f:
            pkl.dump(corpus, f)
            log.info("saved corpus pkl")
    log.info(f"corpus length: {len(corpus)}")
    return corpus


# def make_2wikimultihopqa_documents(elasticsearch_index: str, metadata: Dict = None):
#     raw_filepaths = [
#         os.path.join("raw_data", "2wikimultihopqa", "train.json"),
#         os.path.join("raw_data", "2wikimultihopqa", "dev.json"),
#         os.path.join("raw_data", "2wikimultihopqa", "test.json"),
#     ]
#     metadata = metadata or {"idx": 1}
#     assert "idx" in metadata

#     used_full_ids = set()
#     for raw_filepath in raw_filepaths:

#         with open(raw_filepath, "r") as file:
#             full_data = json.load(file)
#             for instance in tqdm(full_data):

#                 for paragraph in instance["context"]:

#                     title = paragraph[0]
#                     paragraph_text = " ".join(paragraph[1])
#                     paragraph_index = 0
#                     url = ""
#                     is_abstract = paragraph_index == 0

#                     full_id = hash_object(" ".join([title, paragraph_text]))
#                     if full_id in used_full_ids:
#                         continue

#                     used_full_ids.add(full_id)
#                     id_ = full_id[:32]

#                     es_paragraph = {
#                         "id": id_,
#                         "title": title,
#                         "paragraph_index": paragraph_index,
#                         "paragraph_text": paragraph_text,
#                         "url": url,
#                         "is_abstract": is_abstract,
#                     }
#                     document = {
#                         "_op_type": "create",
#                         "_index": elasticsearch_index,
#                         "_id": metadata["idx"],
#                         "_source": es_paragraph,
#                     }
#                     yield (document)
#                     metadata["idx"] += 1
