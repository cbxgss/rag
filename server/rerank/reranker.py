import dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel


dotenv.load_dotenv("../../.env")


class BgeReranker:
    def __init__(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        print(f"Using device: {self.device}")
        self.model_path = "BAAI/bge-reranker-v2-m3"
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=True).half().to(self.device)
        if torch.cuda.device_count() > 1:
            self.rerank_model = torch.nn.DataParallel(self.rerank_model)
        self.rerank_model.eval()


    def rerank(self, query: str, documents: list[str]) -> list[float]:
        pairs = [[query, d] for d in documents]
        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs, padding=True, truncation=True, return_tensors='pt', max_length=512
            ).to(self.device)
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
        return scores


bge_reranker = BgeReranker()

app = FastAPI()


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


@app.post("/rerank/")
async def rerank(item: RerankRequest):
    scores = bge_reranker.rerank(item.query, item.documents)
    return scores
