import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dotenv
from copy import deepcopy
import json
from tqdm.asyncio import tqdm as atqdm
import asyncio

from src.startup.builder import RunBuilder, RAGBuilder
from src.finetune import Data2
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
    async def entity_repeat(entities: list[str], entities_gt: list[str]) -> dict[str, str]:
        # entities 应该与 entities_gt 一一对应，但现在 entities 中有重复的 entity (需要用 rerank 判断是否重复)
        gt2query = {}       # 特别的 "": v, v 代表无效的 query
        for entity in entities:
            entity_gt = await Extractor.entity_match(entity, entities_gt)
            if entity_gt is None:
                gt2query.setdefault("", []).append((entity, 0))
            else:
                gt2query.setdefault(entity_gt[0], []).append((entity, entity_gt[1]))
        # 只保留 "" 和 len(v) > 1 的
        gt2query = {k: v for k, v in gt2query.items() if k == "" or len(v) > 1}
        gt2query_max = {k: max(v, key=lambda x: x[1]) for k, v in gt2query.items() if k != ""}
        gt2query_max = {k: v[0] for k, v in gt2query_max.items()}

        # query2querygt: v 为 score 最高的, k 为要被替换的 query
        query2querygt = {}
        for k, vs in gt2query.items():
            if k == "":
                for v, _ in vs:
                    query2querygt[v] = ""
            else:
                for v, _ in vs:
                    query2querygt[v] = gt2query_max[k]
        return query2querygt

    @staticmethod
    def entity_repeat_trace_pros(trace_step: dict, query2querygt: dict[str, str]):
        knowledge_bak = deepcopy(trace_step["knowledge"])
        for query in knowledge_bak.keys():
            if query in query2querygt:
                querygt = query2querygt[query]
                if querygt == "":
                    trace_step["knowledge"].pop(query)
                elif querygt == query:
                    continue
                elif querygt not in trace_step["knowledge"]:
                    trace_step["knowledge"].pop(query)
                    trace_step["knowledge"][querygt] = knowledge_bak[query]
                else:
                    for doc in knowledge_bak[query]:
                        if doc not in trace_step["knowledge"][querygt]:
                            trace_step["knowledge"][querygt].append(doc)
                    trace_step["knowledge"].pop(query)
        if "retrive" in trace_step:
            retrive_bak = deepcopy(trace_step["retrive"])
            for query in retrive_bak.keys():
                if query in query2querygt:
                    querygt = query2querygt[query]
                    if querygt != query:
                        trace_step["retrive"].pop(query)

    @staticmethod
    async def extract_log(dataset: str, path: str) -> tuple[list[Data2], int]:
        data_li, candidate = [], 0
        with open(path, "r") as f:
            log = json.load(f)
            question = log["question"]
            evaluate = log["evaluate"][0]
            if evaluate == 1:
                trace = log["trace"]
                entity_gt_li = Extractor.extract_entity_gt(dataset, log["metadata"])
                # judge 是否检索多余: retrieve_entity 与 entity_gt_li 一一配对 (用 rerank 配对, 如果没有完全匹配则 candidate = True)
                retrieve_entity = [entity for step_i, trace_step in trace["trace"].items() if "retrive" in trace_step for entity in trace_step["retrive"].keys()]
                entity_gt_li_tmp = deepcopy(entity_gt_li)
                for entity in retrieve_entity:
                    if len(entity_gt_li_tmp) == 0:
                        candidate = 1
                        break
                    entity_gt = await Extractor.entity_match(entity, entity_gt_li_tmp)
                    if entity_gt is None:
                        candidate = 1
                        break
                    entity_gt_li_tmp.remove(entity_gt[0])
                # entity 去重映射
                if candidate == 1:
                    query2querygt = await Extractor.entity_repeat(retrieve_entity, entity_gt_li)
                    for step_i, trace_step in trace["trace"].items():
                        Extractor.entity_repeat_trace_pros(trace_step, query2querygt)
                metadata = {} if candidate == 0 else {"query2querygt": query2querygt}
                # extract
                _know_entity = set()
                for step_i, trace_step in trace["trace"].items():
                    if "retrive" in trace_step:
                        knowledge = trace_step["knowledge"]
                        thought = trace["thought"][:int(step_i) + 1]
                        known_entity = list(_know_entity)
                        rsp = {}
                        for entity in trace_step["retrive"].keys():
                            rsp[entity] = list(trace_step["retrive"][entity]["r"].keys())
                            _know_entity.add(entity)
                        response = {
                            "entities": [
                                {
                                    "entity": entity,
                                    "keywords": keywords
                                } for entity, keywords in rsp.items()
                            ]
                        }
                        data_li.append(Data2(knowledge, question, thought, known_entity, response, entity_gt_li, metadata))
            return data_li, candidate

    @staticmethod
    async def extract_logs_batch(dataset: str, path_batch: list[str]):
        data_li, candidate_li = [], []
        results = await asyncio.gather(*[Extractor.extract_log(dataset, p) for p in path_batch])
        for data_batch, candidate in results:
            if candidate == 0:
                data_li.extend(data_batch)
            else:
                candidate_li.extend(data_batch)
        return data_li, candidate_li

    @staticmethod
    async def extract_logs_all(dataset: str, path: str):
        json_path = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(".json")]
        data_all, candidate_all = [], []
        bs = 20
        tbar = atqdm(range(0, len(json_path), bs), desc="Extracting logs")
        for i in tbar:
            l, r = i, min(i + bs, len(json_path))
            data_batch, candidate_batch = await Extractor.extract_logs_batch(dataset, json_path[l:r])
            data_all.extend(data_batch)
            candidate_all.extend(candidate_batch)
            tbar.set_postfix(data=len(data_all), candidate=len(candidate_all))
        return data_all, candidate_all


def main():
    for dataset, path_log in path_logs.items():
        print(f"Extracting logs from {dataset}...")
        path = f"log/rag/sft-data/{dataset}/{path_log}/output"
        data_all, candidate_all = asyncio.run(Extractor.extract_logs_all(dataset, path))

        os.makedirs("tmp/sft/2", exist_ok=True)
        Data2.save_li(data_all, f"tmp/sft/2/{dataset}_data_dict.json")
        Data2.save_li(candidate_all, f"tmp/sft/2/{dataset}_candidate_dict.json")


if __name__ == "__main__":
    main()
