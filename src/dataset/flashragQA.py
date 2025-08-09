import json
from src.dataset import Item


def get_flashrag_qa(jsonl_path: str) -> list[Item]:
    with open(jsonl_path,"r",encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
        data: list[Item] = [
            Item(
                id=item["id"], question=item["question"], golden_answers=item["golden_answers"],
                metadata=item["metadata"] if "metadata" in item else {}
            )
            for item in items
        ]
        return data
