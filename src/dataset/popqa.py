from src.dataset import Dataset
from src.dataset.flashragQA import get_flashrag_qa


def get_popqa():
    dataset_path = "download/data/cbxgss/rag/popqa/test.jsonl"
    data = get_flashrag_qa(dataset_path)
    dataset = Dataset(name="popqa", data=data)
    reanswer = """### 背景说明

我将给你一个问题和对应的回答，现在，我将将答案与标准答案进行对比打分。

由于打分的效果与回答的表达有关，所以在打分之前，需要将原始的回答进行处理。

### 任务描述

你需要认真分析问题和你的回答，将回答转换成标准形式。

为了让你理解何为标准的回答形式，下面是大量问题的答案的形式说明：

- 30% 的答案是人名，例如 King Edward VII, Rihanna
- 13% 的答案是组织名，例如 Cartoonito, Apalachee
- 10% 的答案是地点，例如 Fort Richardson, California
- 9% 的答案是日期，例如 10th or even 13th century
- 8% 的答案是数，例如 79.92 million, 17
- 8% 的答案是艺术品，例如 Die schweigsame Frau
- 6% 的答案是 Yes/No
- 4% 的答案是形容词，例如 conservative
- 1% 的答案是事件，例如 Prix Benois de la Danse
- 6% 的答案是一些专有名词，例如 Cold War, Laban Movement Analysis
- 5% 的答案是一些常见名词，例如 comedy, both men and women

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
