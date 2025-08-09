import dotenv
from openai.types import Embedding, CreateEmbeddingResponse
import json
from io import BytesIO
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer
import torch
from fastapi import FastAPI
import asyncio


dotenv.load_dotenv("../..")

app = FastAPI()


class BgeEmbedder:
    def __init__(self, model_path="BAAI/bge-large-en-v1.5"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(model_path)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to(self.device)
        else:
            model = model.to(self.device)
        self.model = model.eval()

    async def create_embedding(self, texts: list[str], *args, **kwargs):
        with torch.no_grad():
            embeddings = await asyncio.to_thread(self.model.module.encode, texts, *args, **kwargs) \
                if isinstance(self.model, torch.nn.DataParallel) \
                else await asyncio.to_thread(self.model.encode, texts, *args, **kwargs)
            prompt_tokens = sum([len(sentence.split()) for sentence in texts])
            total_tokens = prompt_tokens
            return CreateEmbeddingResponse(
                data=[Embedding(embedding=embedding, index=i, object="embedding") for i, embedding in enumerate(embeddings)],
                model=self.model.__class__.__name__,
                object="list",
                usage={"prompt_tokens": prompt_tokens, "total_tokens": total_tokens}
            )

    def get_dim(self):
        return self.model.module.get_sentence_embedding_dimension() if isinstance(self.model, torch.nn.DataParallel) \
            else self.model.get_sentence_embedding_dimension()

    def get_max_seq_length(self):
        return self.model.module.get_max_seq_length() if isinstance(self.model, torch.nn.DataParallel) \
            else self.model.get_max_seq_length()


print("Loading model...")
embedder = BgeEmbedder(model_path="BAAI/bge-small-en-v1.5")
print("Model loaded")


@app.post("/create_embedding/")
async def create_embedding(corpus: list[str]):
    print(corpus)
    ret = await embedder.create_embedding(corpus)
    return StreamingResponse(
        BytesIO(json.dumps(ret).encode('utf-8')),
        media_type="application/json",
    )


@app.get("/get_dim/")
async def get_dim():
    return embedder.get_dim()


@app.get("/get_max_seq_length/")
async def get_max_seq_length():
    return embedder.get_max_seq_length()
