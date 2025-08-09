import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from src.finetune import Data1, Data2, Data3


datas = {
    1: {
        "hotpotqa": [
            "tmp/sft/1/hotpotqa_data_dict.json",
            "tmp/sft/1/hotpotqa_candidate_dict.json",
        ],
        "2wikimultihopqa": [
            "tmp/sft/1/2wikimultihopqa_data_dict.json",
            "tmp/sft/1/2wikimultihopqa_candidate_dict.json",
        ],
        "musique": [
            "tmp/sft/1/musique_data_dict.json",
            "tmp/sft/1/musique_candidate_dict.json",
        ],
    },
    2: {
        "hotpotqa": [
            "tmp/sft/2/hotpotqa_data_dict.json",
            "tmp/sft/2/hotpotqa_candidate_dict.json",
        ],
        "2wikimultihopqa": [
            "tmp/sft/2/2wikimultihopqa_data_dict.json",
            "tmp/sft/2/2wikimultihopqa_candidate_dict.json",
        ],
        "musique": [
            "tmp/sft/2/musique_data_dict.json",
            "tmp/sft/2/musique_candidate_dict.json",
        ],
    },
    3: {
        "hotpotqa": [
            "tmp/sft/3/hotpotqa_data_dict.json",
            "tmp/sft/3/hotpotqa_candidate_dict.json",
            "tmp/sft/3/hotpotqa_neg_dict.json",
        ],
        "2wikimultihopqa": [
            "tmp/sft/3/2wikimultihopqa_data_dict.json",
            "tmp/sft/3/2wikimultihopqa_candidate_dict.json",
            "tmp/sft/3/2wikimultihopqa_neg_dict.json",
        ],
        "musique": [
            "tmp/sft/3/musique_data_dict.json",
            "tmp/sft/3/musique_candidate_dict.json",
            "tmp/sft/3/musique_neg_dict.json",
        ],
    },
}


def dict2sharegpt(data_i: int, paths: list[str]):
    data_raw = []
    for path in paths:
        with open(path, "r") as f:
            data_raw.extend(json.load(f))
    data_sft = []
    for data in data_raw:
        Data = [Data1, Data2, Data3][data_i - 1]
        item = Data.from_json(data).to_sft()
        data_sft.append({
            "messages": [
                {
                    "role": "user",
                    "content": item["prompt"]
                },
                {
                    "role": "assistant",
                    "content": item["answer"]
                }
            ]
        })
    return data_sft


def save(path: str, data_i: int, data: list[dict]):
    name = ["rag_ft", "infer", "entity_identify", "learn"][data_i]
    # sharegpt format OpenAI
    dataset_info = {
        f"{name}": {
            "file_name": f"rag/{path}/{name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system"
            }
        }
    }
    os.makedirs(f"tmp/sft/{path}", exist_ok=True)
    with open(f"tmp/sft/{path}/{data_i}_dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    # data
    with open(f"tmp/sft/{path}/{name}.json", "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    tmp = {
        "train-all": ["hotpotqa", "2wikimultihopqa", "musique"],
        "train-hotpotqa": ["hotpotqa"],
        "train-2wikimultihopqa": ["2wikimultihopqa"],
        "train-musique": ["musique"],
    }
    for k, v in tmp.items():
        data_all = []
        for data_i, paths in datas.items():
            data = dict2sharegpt(data_i, [path for dataset in v for path in paths[dataset]])
            save(k, data_i, data)
            data_all.extend(data)
        save(k, 0, data_all)
