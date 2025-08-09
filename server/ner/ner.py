import dotenv
from fastapi import FastAPI
import nltk
from transformers import pipeline


dotenv.load_dotenv("../..")


class NER:
    def __init__(self, method="hf") -> None:
        self.method = method
        # nltk
        # nltk.download('punkt_tab')
        # nltk.download('punkt')
        # nltk.download('averaged_perceptron_tagger')
        # nltk.download('averaged_perceptron_tagger_eng')
        # nltk.download('maxent_ne_chunk')
        # nltk.download('maxent_ne_chunker_tab')
        # nltk.download('words')
        # hf
        if self.method == "hf":
            path = "dbmdz/bert-large-cased-finetuned-conll03-english"
            self.ner_pipeline = pipeline(task="ner", model=path, tokenizer=path, aggregation_strategy="simple", device="cuda")

    @staticmethod
    def extract_entities_nltk(text: str):
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        tree = nltk.ne_chunk(tagged)
        entities, types = dict(), set()
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                entity = ' '.join(word for word, _ in subtree.leaves())
                entity_type = subtree.label()
                entities[entity] = entity_type
                types.add(entity_type)
        return entities.keys()

    def extract_entities_hf(self, text: str):
        entities = self.ner_pipeline(text)
        entities = [entity["word"] for entity in entities if entity["score"] >= 0.9]
        return entities

    def extract_entities(self, text: str):
        if self.method == "nltk":
            return self.extract_entities_nltk(text)
        elif self.method == "hf":
            return self.extract_entities_hf(text)
        else:
            raise NotImplementedError


ner = NER()

app = FastAPI()


@app.post("/ner/")
async def aner(params: dict) -> list[str]:
    entities = ner.extract_entities(params["text"])
    return entities
