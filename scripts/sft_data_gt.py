import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dotenv
from copy import deepcopy
import json
from tqdm.asyncio import tqdm as atqdm
import asyncio

from src.finetune import Data1, Data2, Data3
from src.tools import arerank, LLMAgent, CostManagers


dotenv.load_dotenv()

dataset = "2wikimultihopqa"
path_log = "0106_105422-hrag"


class Extractor:
    @staticmethod
    def extract_entity_gt(metadata: dict) -> list[str]:
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
    async def extract_log(path: str) -> tuple[list[Data1], list[Data2], list[Data3]]:
        # TODO
        pass

    @staticmethod
    async def extract_logs_batch(path_batch: list[str]):
        data1_li, data2_li, data3_li = [], [], []
        results = await asyncio.gather(*[Extractor.extract_log(p) for p in path_batch])
        for data1_item, data2_item, data3_item in results:
            data1_li.extend(data1_item)
            data2_li.extend(data2_item)
            data3_li.extend(data3_item)
        return data1_li, data2_li, data3_li

    @staticmethod
    async def extract_logs_all(path: str):
        json_path = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(".json")]
        data1_all, data2_all, data3_all = [], [], []
        bs = 20
        tbar = atqdm(range(0, len(json_path), bs), desc="Extracting logs")
        for i in tbar:
            l, r = i, min(i + bs, len(json_path))
            data1_batch, data2_batch, data3_batch = await Extractor.extract_logs_batch(json_path[l:r])
            data1_all.extend(data1_batch)
            data2_all.extend(data2_batch)
            data3_all.extend(data3_batch)
            tbar.set_postfix({"data1": len(data1_all), "data2": len(data2_all), "data3": len(data3_all)})
        return data1_all, data2_all, data3_all


def main():
    path = f"log/rag/{dataset}/train/{path_log}/output"
    data1_all, data2_all, data3_all = asyncio.run(Extractor.extract_logs_all(path))

    os.makedirs("tmp/sft", exist_ok=True)
    Data1.save_li(data1_all, "tmp/sft/1_gt_dict.json")
    Data2.save_li(data2_all, "tmp/sft/2_gt_dict.json")
    Data3.save_li(data3_all, "tmp/sft/3_gt_dict.json")


if __name__ == "__main__":
    main()
