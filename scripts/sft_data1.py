import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dotenv
from copy import deepcopy
import json
from tqdm.asyncio import tqdm as atqdm
import asyncio

from src.startup.builder import RunBuilder, RAGBuilder
from src.finetune import Data1
from src.tools import arerank, LLMAgent, CostManagers


dotenv.load_dotenv()

path_logs = {
    "hotpotqa": "0131_081938-hrag",
    "2wikimultihopqa": "0106_105422-hrag",
    "musique": "0131_081451-hrag",
}

class Extractor:
    @staticmethod
    def extract_entity_gt(dataset:str, metadata: dict) -> list[str]:
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
    async def entity_gt_known(entities: list[str], entity_gt_li: list[str]) -> bool:
        # 检查是否全部 entity_gt_li 已经被知晓
        gt_known = {entity: False for entity in entity_gt_li}
        for entity in entities:
            entity_gt = await Extractor.entity_match(entity, entity_gt_li)
            if entity_gt is not None:
                gt_known[entity_gt[0]] = True
        return all(gt_known.values())

    @staticmethod
    async def extract_log(dataset:str, path: str) -> tuple[list[Data1], list[Data1]]:
        data_li, candidate_li = [], []
        with open(path, "r") as f:
            log = json.load(f)
            question = log["question"]
            evaluate = log["evaluate"][0]
            if evaluate == 1:
                trace = log["trace"]
                entity_gt_li = Extractor.extract_entity_gt(dataset, log["metadata"])
                # extract
                _know_entity = set()
                for step_i, trace_step in trace["trace"].items():
                    knowledge = trace_step["knowledge"]
                    thought = trace["thought"][:int(step_i)]
                    response = trace_step["thought"]
                    entities_known = list(knowledge.keys())
                    if await Extractor.entity_gt_known(entities_known, entity_gt_li):
                        if response["need_retrieve"]:
                            candidate_li.append(Data1(knowledge, question, thought, response))
                        else:
                            data_li.append(Data1(knowledge, question, thought, response))
                    else:
                        data_li.append(Data1(knowledge, question, thought, response))
            return data_li, candidate_li

    @staticmethod
    async def extract_logs_batch(dataset: str, path_batch: list[str]):
        data_li, candidate_li = [], []
        results = await asyncio.gather(*[Extractor.extract_log(dataset, p) for p in path_batch])
        for data_batch, candidate_batch in results:
            data_li.extend(data_batch)
            candidate_li.extend(candidate_batch)
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

        os.makedirs("tmp/sft/1", exist_ok=True)
        Data1.save_li(data_all, f"tmp/sft/1/{dataset}_data_dict.json")
        Data1.save_li(candidate_all, f"tmp/sft/1{dataset}_candidate_dict.json")


if __name__ == "__main__":
    main()
