from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, corpus: List[str], *args, **kwargs):
        return (
            self.client.embeddings.create(input=corpus, model=self.model).data[0].embedding
        )

    def get_dim(self):
        return 1536
