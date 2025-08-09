import os
from src.dataset.dataset import Item, Dataset
from src.dataset.flashragQA import get_flashrag_qa


def get_2wikimultihopqa():
    dataset_path = os.path.join(
        os.getenv("HF_HOME"), "hub", "datasets--cbxgss--rag", "snapshots", "64d4a872814da55c8284f5536795df03c39ddad2",
        "2wikimultihopqa", "dev.jsonl"
    )
    data = get_flashrag_qa(dataset_path)
    dataset = Dataset(name="2wikimultihopqa", data=data)
    reanswer = """### 背景说明

我将给你一个问题和对应的回答，现在，我将将答案与标准答案进行对比打分。

由于打分的效果与回答的表达有关，所以在打分之前，需要将原始的回答进行处理。

### 任务描述

你需要认真分析问题和你的回答，将回答转换成标准形式。

为了让你理解何为标准的回答形式，下面是大量问题的答案的形式说明：

- 31.2% 的答案是 yes/no
- 16.9% 的答案是日期，例如 July 4, 1776
- 13.5% 的答案是电影名，例如 La La Land
- 11.7% 的答案是人名，例如 George Washington
- 4.7% 的答案是地点，例如 New York City
- 剩余的 22% 的答案往往是 wikidata 中的某个实体

上面是对整个数据集答案的分布说明，下面是一些具体的问题回答的转换示例：

Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Arthur's Magazine

Q: The Oberoi family is part of a hotel company that has a head office in what city?
A: Delhi

Q: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?
A: Richard Milhous Nixon

### 当前问题和答案

现在，认真阅读问题和原始回答，转化原始的回答成标准答案的形式。

问题:

{question}

原始回答:

{response}

### 输出格式

直接输出新的回答，不要输出任何其他内容。
"""
    return dataset, reanswer
