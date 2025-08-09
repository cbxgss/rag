import os
import logging
from tqdm import tqdm
from omegaconf import DictConfig
import json
import tarfile
import bz2
import pickle as pkl


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
            "retrieval-corpus", "hotpotqa", "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
        )
        if not os.path.exists(base_path):
            raise Exception("not exist hotpotqa corpus")
        corpus = []
        with tarfile.open(base_path, "r:bz2") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and os.path.basename(m.name).startswith("wiki_")]
            tbar1 = tqdm(members)
            for member in tbar1:
                tbar1.set_postfix(file=member.name)
                with bz2.open(tar.extractfile(member), "rt") as f:
                    tbar2 = tqdm(f, desc="Reading lines", leave=False)
                    for line in tbar2:
                        instance = json.loads(line.strip())
                        title = instance["title"]
                        sentences_text = [e.strip() for e in instance["text"]]
                        paragraph_text = " ".join(sentences_text)
                        doc = f"#### {title}\n\n{paragraph_text}"
                        corpus.append(doc)
                        tbar1.set_postfix(len=len(corpus))
        with open(pkl_path, "wb") as f:
            pkl.dump(corpus, f)
            log.info("saved corpus pkl")
    return corpus


# def make_hotpotqa_documents(elasticsearch_index: str, metadata: Dict = None):
#     raw_glob_filepath = os.path.join("raw_data", "hotpotqa", "wikpedia-paragraphs", "*", "wiki_*.bz2")
#     metadata = metadata or {"idx": 1}
#     assert "idx" in metadata
#     for filepath in tqdm(glob.glob(raw_glob_filepath)):
#         for datum in bz2.BZ2File(filepath).readlines():
#             instance = json.loads(datum.strip())

#             id_ = hash_object(instance)[:32]
#             title = instance["title"]
#             sentences_text = [e.strip() for e in instance["text"]]
#             paragraph_text = " ".join(sentences_text)
#             url = instance["url"]
#             is_abstract = True
#             paragraph_index = 0

#             es_paragraph = {
#                 "id": id_,
#                 "title": title,
#                 "paragraph_index": paragraph_index,
#                 "paragraph_text": paragraph_text,
#                 "url": url,
#                 "is_abstract": is_abstract,
#             }
#             document = {
#                 "_op_type": "create",
#                 "_index": elasticsearch_index,
#                 "_id": metadata["idx"],
#                 "_source": es_paragraph,
#             }
#             yield (document)
#             metadata["idx"] += 1
