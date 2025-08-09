import dotenv
from fastapi import FastAPI
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


dotenv.load_dotenv("../..")


class NLI:
    def __init__(self) -> None:
        self.nli_tokenizer = AutoTokenizer.from_pretrained("google/t5_xxl_true_nli_mixture", use_fast=False)
        self.nli_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5_xxl_true_nli_mixture", torch_dtype=torch.bfloat16, device_map="auto")

    async def run(self, premise: str, hypothesis: str) -> bool:
        input_text = f"premise: {premise} hypothesis: {hypothesis}"
        input_ids = (await asyncio.to_thread(self.nli_tokenizer, input_text, return_tensors="pt", max_length=1024)).input_ids.to("cuda")
        with torch.no_grad():
            outputs = await asyncio.to_thread(self.nli_model.generate, input_ids, max_new_tokens=10)
        result = await asyncio.to_thread(self.nli_tokenizer.decode, outputs[0], skip_special_tokens=True)
        return result == "1"

nli = NLI()
app = FastAPI()


@app.post("/nli/")
async def aner(params: dict) -> bool:
    result = await nli.run(params["premise"], params["hypothesis"])
    return result
