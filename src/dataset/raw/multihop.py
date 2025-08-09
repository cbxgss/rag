import os
import logging
from omegaconf import DictConfig
import json
import pickle as pkl


log = logging.getLogger(__name__)


def trans_corpus(cfg: DictConfig):
    log.info("loading corpus_raw json")
    with open(cfg.dataset.corpus.path_raw, 'r') as f:
        datasets = json.load(f)
        corpus = [item["body"] for item in datasets]
    with open(cfg.dataset.corpus.path, 'w') as f:
        corpus_json = {str(i): {"title": "", "text": item} for i, item in enumerate(corpus)}
        json.dump(corpus_json, f, indent=4)
    os.makedirs(cfg.corpus_dir, exist_ok=True)
    corpus_path = os.path.join(cfg.corpus_dir, "corpus.pkl")
    with open(corpus_path, 'wb') as f:
        pkl.dump(corpus, f)
    return corpus


def trans_qa(cfg: DictConfig):
    log.info(f"loading qa_raw {cfg.task.dataset}")

    with open(cfg.dataset.qa_path_raw, 'r') as f:
        qa_raw = json.load(f)
    with open(cfg.dataset.corpus.path_raw, 'r') as f:
        corpus_raw = json.load(f)
        url2corpusid = {doc["url"]: i for i, doc in enumerate(corpus_raw)}
    qa = []
    for item in qa_raw:
        url = [doc["url"] for doc in item["evidence_list"]]
        corpusid = [url2corpusid[u] for u in url]
        line = {
            "id": str(len(qa)),
            "question": item["query"],
            "golden_answers": [item["answer"]],
            "metadata": {
                "doc_id": corpusid,
                "doc": [corpus_raw[i]["body"] for i in corpusid]
            }
        }
        qa.append(line)

    with open(cfg.dataset.qa_path, 'w') as f:
        for item in qa:
            f.write(json.dumps(item) + "\n")
    return qa
