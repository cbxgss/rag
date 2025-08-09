import logging
import json
from dataclasses import dataclass
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizer

from src.startup.builder import RunBuilder, RAGBuilder
from src.rag.duralrag.prompt import *


log = logging.getLogger(__name__)

@dataclass
class Data1:
    knowledge: dict[str, list[str]]
    question: str
    thought: list[str]
    response: dict

    prompt_template = """## Background Information

Currently, there is a knowledge question-answering problem that needs to be solved, and retrieval tools can be used to find relevant knowledge to assist you in solving the problem.

For knowledge question-answering tasks, it is important to have a grasp of knowledge, which is organized by entities, with each entity having a segment of knowledge.

## Task Description

Your current task is to reason based on the knowledge that has been retrieved.

When reasoning, you must strictly adhere to the knowledge that has been retrieved to prevent errors.

There are two conditions for concluding your reasoning:
- You have obtained the answer you were seeking, at which point you can conclude your reasoning.
- You find that you cannot obtain the answer you want based solely on the retrieved knowledge and need to further expand your knowledge through retrieval tools.

## Note

- You must reason step by step carefully to ensure the rigor of the reasoning process.
- The knowledge used must be strictly based on the retrieved knowledge, and speculation is prohibited.

## Question currently being solved

### Known Knowledge

{knowledge}

### Current Knowledge Question

{question}

### Reasoning History

{thought}

### Output
"""

    @staticmethod
    def knowledge2str(knowledge: dict):
        def item2str(entity, content):
            return f"#### {entity}\n\n" + "\n\n".join(content)
        return "\n\n".join([item2str(entity, content) for entity, content in knowledge.items()])

    def to_json(self):
        return {
            "knowledge": self.knowledge,
            "question": self.question,
            "thought": self.thought,
            "response": self.response,
        }

    @staticmethod
    def from_json(data):
        return Data1(
            data["knowledge"],
            data["question"],
            data["thought"],
            data["response"]
        )

    def prompt(self):
        return Data1.prompt_template.format(
            knowledge=Data1.knowledge2str(self.knowledge),
            question=self.question,
            thought="\n".join([f"{i + 1}. {t}" for i, t in enumerate(self.thought)]),
        )

    def answer(self):
        return json.dumps(self.response, indent=4, ensure_ascii=False)

    def to_sft(self):
        return {
            "prompt": self.prompt(),
            "answer": self.answer()
        }

    @staticmethod
    def save_li(data_li: list["Data1"], path: str):
        with open(path, "w") as f:
            json.dump([data.to_json() for data in data_li], f, indent=4, ensure_ascii=False)


@dataclass
class Data2:
    knowledge: dict[str, list[str]]
    question: str
    thought: list[str]
    known_entity: list[str]
    response: dict
    entity_gts: list[str]
    metadata: dict

    prompt_template = """## Background

Currently, there is a knowledge question that needs to be solved, and a retrieval tool can be used to find relevant knowledge to assist you in resolving the issue.

For knowledge question tasks, it is important to have a grasp of the knowledge, which is organized by entities, each of which has a segment of knowledge.

### Task Description

Your current task is to **identify what additional knowledge is needed** based on the given question, the existing knowledge, and the previous reasoning history, and to **generate retrieval keywords**.

**Identify What Additional Knowledge is Needed**

A knowledge point is a key piece of information necessary to solve the current problem. It often revolves around a noun-like entity, which can be a person, location, organization, event, or proper noun.

To help you identify the required knowledge, I will extract a list of entities from previous reasoning processes. These entities can help you pinpoint key knowledge points. They may not all be accurate, but they are generally helpful for guidance.

**Generate Retrieval Keywords**

- The generated retrieval keywords will be used by a dense retrieval tool. The keywords should meet the requirements of this tool to ensure relevant documents are retrieved.
- For the same knowledge point, it may be necessary to retrieve multiple sub-knowledge points. Ensure that the generated retrieval keywords cover all the required sub-knowledge points. However, focus only on the knowledge points relevant to the current question and avoid excessive retrieval.
- For a single sub-knowledge point, to improve the recall of relevant documents, you may need multiple retrieval keywords with the same meaning but different expressions. However, for similar-meaning keywords, retain at most **two variations**.

## Question currently being solved

### Known Knowledge

The knowledge that has been retrieved is as follows:

{knowledge}

### Question

The original question is as follows:

{question}

### Reasoning History

Your previous reasoning history is as follows:

{thought}

### Hint Entities

If you need to continue retrieving information on a previously generated knowledge point, ensure that the name of that specific knowledge point remains consistent. The knowledge points that have been retrieved are as follows:

{known_entity}

Use these entities as a reference to identify key knowledge points and generate retrieval keywords accordingly.

If the necessary knowledge points are unclear from the start, you may generate an additional knowledge point named "else" to store retrieval queries for such ambiguous knowledge.

### Output
"""

    @staticmethod
    def knowledge2str(knowledge: dict):
        def item2str(entity, content):
            return f"#### {entity}\n\n" + "\n\n".join(content)
        return "\n\n".join([item2str(entity, content) for entity, content in knowledge.items()])

    def to_json(self):
        return {
            "knowledge": self.knowledge,
            "question": self.question,
            "entity_gts": self.entity_gts,
            "metadata": self.metadata,
            "thought": self.thought,
            "known_entity": self.known_entity,
            "response": [self.response]
        }

    @staticmethod
    def from_json(data):
        return Data2(
            data["knowledge"],
            data["question"],
            data["thought"],
            data["known_entity"],
            data["response"][0],
            data["entity_gts"],
            data["metadata"]
        )

    def prompt(self):
        return Data2.prompt_template.format(
            knowledge=Data2.knowledge2str(self.knowledge),
            question=self.question,
            thought="\n".join([f"{i + 1}. {t}" for i, t in enumerate(self.thought)]),
            known_entity=", ".join(self.known_entity)
        )

    def answer(self):
        return json.dumps(self.response, indent=4, ensure_ascii=False)

    def to_sft(self):
        return {
            "prompt": self.prompt(),
            "answer": self.answer()
        }

    @staticmethod
    def save_li(data_li: list["Data2"], path: str):
        with open(path, "w") as f:
            json.dump([data.to_json() for data in data_li], f, indent=4, ensure_ascii=False)


@dataclass
class Data3:
    question: str
    thought: list[str]
    key_entity: str
    query: list[str]
    docs: list[str]
    response: str
    entity_gts: list[str]

    prompt_template = """## Task Description

You are assisting in solving a QA problem, and you have gathered relevant information using retrieval tools.

Your task is to read and organize the retrieved documents, filtering out irrelevant content while summarizing information pertinent to the current problem. When assessing the usefulness of the content, consider that some information may not appear directly related to the final answer but could be essential for multi-hop reasoning. Even if content does not lead to an immediate conclusion, it may provide necessary context or intermediary insights that help progress toward the answer.

## Note

- Summarize the content directly without adding personal commentary or interpretations. Do not infer or speculate about missing information.
- Preserve the original wording for important content and **ensure that all entity names remain consistent with the original documents**.

## Question currently being solved

### Original Question

{question}

### Reasoning History

{thought}

### Retrieval

Focus on extracting and summarizing information that relates to both the key entities and the aspects highlighted in the retrieve query. Emphasize connections that could facilitate multi-hop reasoning, ensuring that no potentially useful information is overlooked simply because it does not directly lead to the final answer.

The key entity, query, and retrieved content are provided below:

Key Entity Retrieved: {entity}

Retrieve Queries: {query}

Retrieved Content:

{docs}

### Output
"""

    def to_json(self):
        return {
            "question": self.question,
            "entity_gts": self.entity_gts,
            "thought": self.thought,
            "key_entity": self.key_entity,
            "query": self.query,
            "docs": self.docs,
            "response": self.response,
        }

    @staticmethod
    def from_json(data):
        return Data3(
            data["question"],
            data["thought"],
            data["key_entity"],
            data["query"],
            data["docs"],
            data["response"],
            data["entity_gts"]
        )

    def prompt(self):
        return Data3.prompt_template.format(
            question=self.question,
            thought="\n".join([f"{i + 1}. {t}" for i, t in enumerate(self.thought)]),
            entity=self.key_entity,
            query=self.query,
            docs="\n\n".join(self.docs)
        )

    def answer(self):
        return self.response

    def to_sft(self):
        return {
            "prompt": self.prompt(),
            "answer": self.answer()
        }

    @staticmethod
    def save_li(data_li: list["Data3"], path: str):
        with open(path, "w") as f:
            json.dump([data.to_json() for data in data_li], f, indent=4, ensure_ascii=False)
