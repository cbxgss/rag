import sys
sys.path.append(".")
import os
import json
import pandas as pd

from scripts.logs import logs
from src.corpus import  hash_object
from src.evaluator import  retrieval_metrics

dataset = "2wikimultihopqa" # "nq", "eli5", "asqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle", "strategyqa"
print(f"dataset: {dataset}")
k_values = [3, 10, 50, 200]
methods = [
    "native",
    "ircot",
    "hrag",
    "hrag-mem",
    "lazykrag",
]


def get_docs_retrieved(trace: dict, method: str) -> list[str]:
    docs = set()
    if "native" in method:
        rerank = trace["rerank"]
        docs.update(set([doc for doc_id, doc in rerank["docs"].items() if rerank["scores"][doc_id] > 0]))
    elif "ircot" in method:
        for i in trace:
            rerank = trace[i]["rerank"] if "rerank" in trace[i] else trace[i]["retrieve"]["rerank"]
            docs.update(set([doc for doc_id, doc in rerank["docs"].items() if rerank["scores"][doc_id] > 0]))
    elif "hrag" in method:
        for i in trace["trace"]:
            if "learn" in trace["trace"][i]:
                docs.update(set([doc for v in trace["trace"][i]["learn"].values() for doc in v["new_docs"].values()]))
    elif "lazykrag" in method:
        docs.update(set([doc for doc in trace["docs"].values()]))
    else:
        raise NotImplementedError
    return list(docs)


def eval_one(log_filepath: str, method: str):
    with open(log_filepath) as f:
        log_content = json.load(f)
        content_key = "sentences" if dataset == "hotpotqa" else "content"

        titles: list[str] = log_content["metadata"]["context"]["title"]
        contents: list[list[str]] = log_content["metadata"]["context"][content_key]

        doc_id = [titles.index(title) for title in log_content["metadata"]["supporting_facts"]["title"]]  # 有效 doc id
        sent_id = log_content["metadata"]["supporting_facts"]["sent_id"]                                  # 有效 sentence id

        support_titles = [titles[i] for i in doc_id]
        join_token = "" if dataset == "hotpotqa" else " "
        support_contents: list[str] = [join_token.join(contents[di]) for di in doc_id]
        docs = [f"#### {title}\n\n{content}" for title, content in zip(support_titles, support_contents)]
        docs_id = [hash_object(doc) for doc in docs]

        docs_retrieved = get_docs_retrieved(log_content["trace"], method)
        docs_retrieved_id = [hash_object(doc) for doc in docs_retrieved]

        score = {doc_id: 1 for doc_id in docs_retrieved_id}
        relevance = {doc_id: 1 for doc_id in docs_id}

        # print(json.dumps(docs, indent=4))
        # print(json.dumps(docs_retrieved, indent=4))
        # print(json.dumps(score, indent=4))
        # print(json.dumps(relevance, indent=4))
        return score, relevance


def eval_method_one(log_base_path: str, method: str, cnt: int):
    ids = [f.replace(".json", "") for f in os.listdir(log_base_path) if f.endswith(".json")]
    ret = {f"{cnt}_{id}": eval_one(f"{log_base_path}/{id}.json", method) for id in ids}
    scores = {id: v[0] for id, v in ret.items()}
    relevance = {id: v[1] for id, v in ret.items()}
    return scores, relevance


def eval_method(method: str):
    scores_all = {}
    relevance_all = {}
    for cnt, log_dir in enumerate(logs[dataset][method]):
        log_base_path = f"log/rag/{dataset}/{log_dir}/output"
        scores, relevance = eval_method_one(log_base_path, method, cnt)
        scores_all.update(scores)
        relevance_all.update(relevance)
    metrics = retrieval_metrics(scores_all, relevance_all, k_values)
    return metrics


def eval_dataset():
    # 构建表头
    header_top = ["recall"] * len(k_values) + ["ndcg"] * len(k_values) + ["map"] * len(k_values)
    header_bottom = k_values * 3
    headers = pd.MultiIndex.from_arrays([header_top, header_bottom])

    table_data = []
    for method in methods:
        metrics = eval_method(method)
        row = []
        for metric in ["recall", "ndcg", "map"]:
            for k in k_values:
                key = f"{metric}@{k}"
                row.append(metrics[metric][key])
        table_data.append(row)

    df = pd.DataFrame(table_data, columns=headers, index=methods)
    print(df)


eval_dataset()
