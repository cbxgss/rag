from omegaconf import DictConfig
import logging
from cr_utils import Logger

from src.dataset import Item
from src.rag.native import NativeRAG
from src.startup import RAGBuilder


log = logging.getLogger(__name__)


@RAGBuilder.register_module("oracle")
class Oracle(NativeRAG):
    def __init__(self, cfg: DictConfig, logger: Logger):
        super().__init__(cfg, logger)

    async def aquery(self, data: Item):
        log_dir = f"log/{data.id}/llm"
        response = await self.agenerate(data.question, self.get_context(data), log_dir)
        return response, {}

    def get_context(self, data: Item):
        if self.cfg.dataset == "hotpotqa" or self.cfg.dataset == "2wikimultihopqa":
            content_key = "sentences" if self.cfg.dataset == "hotpotqa" else "content"

            titles: list[str] = data.metadata["context"]["title"]
            contents: list[list[str]] = data.metadata["context"][content_key]

            doc_id = [titles.index(title) for title in data.metadata["supporting_facts"]["title"]]  # 有效 doc id
            sent_id = data.metadata["supporting_facts"]["sent_id"]                                  # 有效 sentence id

            support_titles = [titles[i] for i in doc_id]
            if self.cfg.task.method.mode == "paragraph":
                support_contents: list[str] = ["".join(contents[di]) for di in doc_id]
            elif self.cfg.task.method.mode == "sentence":
                support_contents: list[str] = [contents[di][sent_id[i]] for i, di in enumerate(doc_id)]
            else:
                raise ValueError(f"upper mode: {self.cfg.task.method.mode} not supported")
        elif self.cfg.dataset == "musique":
            support_titles = [sub_q["support_paragraph"]["title"] for sub_q in data.metadata["question_decomposition"]]
            support_contents = [sub_q["support_paragraph"]["paragraph_text"] for sub_q in data.metadata["question_decomposition"]]
        elif self.cfg.dataset == "multihopqa":
            evidence_li = data.metadata["evidence_list"]
            support_titles = [evidence["title"] for evidence in evidence_li]
            support_contents = [evidence["fact"] for evidence in evidence_li]
        else:
            raise ValueError(f"dataset: {self.cfg.dataset} not support Oracle")
        docs = [f"### {title}\n\n{content}" for title, content in zip(support_titles, support_contents)]
        return "\n\n".join(docs)
