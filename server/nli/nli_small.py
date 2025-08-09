import dotenv
from fastapi import FastAPI
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


dotenv.load_dotenv("../../.env")


class NLI:
    def __init__(self) -> None:
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda")

    async def run(self, premise: str, hypothesis: str) -> bool:
        input_text = f"premise: {premise} hypothesis: {hypothesis}"
        input_ids = (await asyncio.to_thread(self.nli_tokenizer, input_text, return_tensors="pt", max_length=1024)).input_ids.to(self.nli_model.device)
        with torch.no_grad():
            outputs = await asyncio.to_thread(self.nli_model, input_ids)
            prediction = torch.softmax(outputs["logits"][0], -1).tolist()
            label_names = ["entailment", "neutral", "contradiction"]
            prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
            return prediction["entailment"] > 50

nli = NLI()
app = FastAPI()


@app.post("/nli/")
async def aner(params: dict) -> bool:
    result = await nli.run(params["premise"], params["hypothesis"])
    return result
