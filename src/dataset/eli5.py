import json
from src.dataset import Item, Dataset


def get_eli5():
    data: list[Item] = []
    dataset_path = "download/data/cbxgss/rag/eli5/dev.jsonl"
    with open(dataset_path,"r",encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
        data: list[Item] = [
            Item(
                id=item["id"], question=item["question"], golden_answers=item["golden_answers"],
                metadata={k: v for k, v in item.items() if k not in ["id", "question", "golden_answers"]}
            )
            for item in items
        ]
    dataset = Dataset(name="eli5", data=data)

    extract_answer_prompt = None
    return dataset, extract_answer_prompt
