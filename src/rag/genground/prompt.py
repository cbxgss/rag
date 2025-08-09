from pydantic import BaseModel


I_A = """### Task Description

Please decompose a multi-hop question into sub-questions and answer the sub-questions step by step.

Starting below, you should interleave Deduce and Answer until deriving at the final answer.
- Deduce: deduce the current context and then formulate a subquestion
- Answer: answer the deduced question

### Example

Your output should be in the following json format.

Here are some examples:

#### Example 1

**Question**: Which magazine was started first Arthur's Magazine or First for Women?

**History**: None

**Output**:

{{
    "subquestion": "When was Arthur's Magazine started?",
    "answer": "1884"
}}

#### Example 2

**Question**: The Oberoi family is part of a hotel company that has a head office in what city?

**History**:

- Item 1:
    - subquestion: "Which hotel company is the Oberoi family part of?"

**Output**:

{{
    "subquestion": "Where is the head office of the hotel company?",
    "answer": "Delhi"
}}

### Question currently being solved

The original question is:

{question}

The previous history of Deduce and Answer is as follows:

{history}
"""

class SchemaA(BaseModel):
    subquestion: str
    answer: str

response_format_A = {
    "type": "json_schema",
    "json_schema": {
        "name": "deduce_answer",
        "schema": SchemaA.model_json_schema(),
    }
}


I_G = """### Task Description

Here is an answer to the question. Please cite evidence from the documents list to revise the answer. You should encapsulate the evidence using "<ref></ref>", and the revised answer using "<revise> </revise>".

If no evidence can be found, just give "<ref> Empty </ref>".

### Question currently being solved

The original question is:

{question}

Sub-question and original answer that need to be grounded:

- subquestion: {subquestion}
- original answer: {answer}

Retrieved document:

{docs}
"""

I_answer = """### Task Description

You are solving a knowledge question, and you have used retrieval tools to obtain reliable answers to some sub-questions.

Now, you need to answer the original question based on these sub-questions and their corresponding answers.

### Original Question

{question}

### Sub-Questions and Answers

{history}
"""
