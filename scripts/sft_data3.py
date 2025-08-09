import os, sys

from markdown_it.rules_inline import entity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dotenv
from copy import deepcopy
import json
from tqdm.asyncio import tqdm as atqdm
import asyncio

from src.startup.builder import RunBuilder, RAGBuilder
from src.finetune import Data3
from src.rag.duralrag.prompt import *
from src.tools import arerank, LLMAgent, CostManagers


dotenv.load_dotenv()

path_logs = {
    "hotpotqa": "0131_081938-hrag",
    "2wikimultihopqa": "0106_105422-hrag",
    "musique": "0131_081451-hrag",
}


class Extractor:
    @staticmethod
    def extract_entity_gt(dataset: str, metadata: dict) -> list[str]:
        if dataset == "hotpotqa":
            return metadata["supporting_facts"]["title"]
        elif dataset == "2wikimultihopqa":
            return metadata["supporting_facts"]["title"]
        elif dataset == "musique":
            return [sub_q["support_paragraph"]["title"] for sub_q in metadata["question_decomposition"]]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    @staticmethod
    async def entity_match(entity: str, entities_gt: list[str]) -> tuple[str, float] | None:
        query = entity
        docs = entities_gt
        docs, scores, idxs = await arerank(query, list(range(len(docs))), docs, k=1, filter=True)
        if len(docs) == 0:
            return None
        else:
            return docs[0], scores[0]

    @staticmethod
    async def extract_log(dataset: str, path: str) -> tuple[list[Data3], list[Data3], list[Data3]]:
        data_li, candidate_li, neg_li = [], [], []
        with open(path, "r") as f:
            log = json.load(f)
            question = log["question"]
            evaluate = log["evaluate"][0]
            if evaluate == 1:
                trace = log["trace"]
                entity_gt_li = Extractor.extract_entity_gt(dataset, log["metadata"])
                # extract
                for step_i, trace_step in trace["trace"].items():
                    if "learn" in trace_step:
                        thought = trace["thought"][:int(step_i) + 1]
                        entity2query = {}
                        for entity in trace_step["retrive"].keys():
                            entity2query[entity] = list(trace_step["retrive"][entity]["r"].keys())
                        for k, v in trace_step["learn"].items():
                            key_entity = k
                            query = entity2query[k]
                            docs = list(v["new_docs"].values())
                            if len(docs) == 0:
                                continue
                            rsp = v["read"]
                            ma = await Extractor.entity_match(key_entity, entity_gt_li)
                            if ma:
                                if "None" in rsp or "none" in rsp:
                                    entity_gt = ma[0]
                                    recall_flag = any([f"#### {entity_gt}" in doc for doc in docs])
                                    if recall_flag:
                                        if dataset == "hotpotqa" or dataset == "2wikimultihopqa":
                                            sent_idx = log["metadata"]["supporting_facts"]["title"].index(entity_gt)
                                            sent_id = log["metadata"]["supporting_facts"]["sent_id"][sent_idx]
                                            doc_idx = log["metadata"]["context"]["title"].index(entity_gt)
                                            k = "content" if dataset == "2wikimultihopqa" else "sentences"
                                            sent = log["metadata"]["context"][k][doc_idx][sent_id]
                                        else:
                                            entity_idx = [i for i, sub_q in enumerate(log["metadata"]["question_decomposition"]) if sub_q["support_paragraph"]["title"] == entity_gt]
                                            entity_idx = entity_idx[0]
                                            sent = log["metadata"]["question_decomposition"][entity_idx]["support_paragraph"]["paragraph_text"]
                                        candidate_li.append(Data3(question, thought, key_entity, query, docs, sent, entity_gt_li))
                                    else:
                                        candidate_li.append(Data3(question, thought, key_entity, query, docs, "None", entity_gt_li))
                                else:
                                    data_li.append(Data3(question, thought, key_entity, query, docs, rsp, entity_gt_li))
                            else:
                                rsp = "None"
                                neg_li.append(Data3(question, thought, key_entity, query, docs, rsp, entity_gt_li))
            return data_li, candidate_li, neg_li

    @staticmethod
    async def extract_logs_batch(dataset: str, path_batch: list[str]):
        data_li, candidate_li, neg_li = [], [], []
        results = await asyncio.gather(*[Extractor.extract_log(dataset, p) for p in path_batch])
        for data_item, candidate_item, neg_item in results:
            data_li.extend(data_item)
            candidate_li.extend(candidate_item)
            neg_li.extend(neg_item)
        return data_li, candidate_li, neg_li

    @staticmethod
    async def extract_logs_all(dataset: str, path: str):
        json_path = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(".json")]
        data_all, candidate_all, neg_all = [], [], []
        bs = 100
        tbar = atqdm(range(0, len(json_path), bs), desc="Extracting logs")
        for i in tbar:
            l, r = i, min(i + bs, len(json_path))
            data_batch, candidate_batch, neg_batch = await Extractor.extract_logs_batch(dataset, json_path[l:r])
            data_all.extend(data_batch)
            candidate_all.extend(candidate_batch)
            neg_all.extend(neg_batch)
            tbar.set_postfix(data=len(data_all), candidate=len(candidate_all), neg=len(neg_all))
        return data_all, candidate_all, neg_all


def main():
    for dataset, path_log in path_logs.items():
        print(f"Extracting logs from {dataset}...")
        path = f"log/rag/sft-data/{dataset}/{path_log}/output"
        data_all, candidate_all, neg_all = asyncio.run(Extractor.extract_logs_all(dataset, path))

        os.makedirs("tmp/sft/3", exist_ok=True)
        Data3.save_li(data_all, f"tmp/sft/3/{dataset}_data_dict.json")
        Data3.save_li(candidate_all, f"tmp/sft/3/{dataset}_candidate_dict.json")
        Data3.save_li(neg_all, f"tmp/sft/3/{dataset}_neg_dict.json")


if __name__ == "__main__":
    main()
