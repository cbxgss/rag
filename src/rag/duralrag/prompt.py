from pydantic import BaseModel


shot_infer = """
## Output Format

You should output in JSON format:

{{
    "thought": "str, your reasoning thoughts",
    "need_retrieve": True / False
}}

## Example

### Example1

**Question**: Which magazine was started first Arthur's Magazine or First for Women?

**Known Knowledge**: None

**Reasoning History**: None

**Output**:

{{
    "thought": "I need to find out the founding dates of Arthur's Magazine and First for Women separately."
    "need_retrieve": True
}}

### Example2

**Question**: The Oberoi family is part of a hotel company that has a head office in what city?

**Known Knowledge**: None

**Reasoning History**: None

**Output**:

{{
    "thought": "I need to find out which hotel company the Oberoi family is part of."
    "need_retrieve": True
}}

### Example3

**Question**: The Oberoi family is part of a hotel company that has a head office in what city?

**Known Knowledge**:

- **Oberoi family**: The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.

**Reasoning History**:
1. I need to find out which hotel company the Oberoi family is part of.

**Output**:

{{
    "thought": "The Oberoi family is part of The Oberoi Group, which is an Indian hotel company. Next, I need to find out what city The Oberoi Group is headquartered in."
    "need_retrieve": True
}}
"""

prompt_infer = """## Background Information

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
""" + shot_infer + """
## Question currently being solved

### Known Knowledge

{knowledge}

### Current Knowledge Question

{question}

### Reasoning History

{thought}

### Output

"""

class SchemaInfer(BaseModel):
    thought: str
    need_retrieve: bool

response_format_infer = {
    "type": "json_schema",
    "json_schema": {
        "name": "ThoughtProcess",
        "schema": SchemaInfer.model_json_schema()
    }
}

# response_format_infer = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "ThoughtProcess",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "thought": {
#                     "type": "string",
#                     "description": "The current reasoning process, describing the thought process or the knowledge needed if the answer is not yet obtained."
#                 },
#                 "need_retrieve": {
#                     "type": "boolean",
#                     "description": "Indicates if additional retrieval or knowledge lookup is required. Set to True if retrieval is needed."
#                 }
#             },
#             "required": ["thought", "need_retrieve"],
#             "additionalProperties": False
#         },
#         "strict": True
#     }
# }

shot_need = """
## Output Format

You should output in JSON format:

{{
    "entities": [
        {{
            "entity": "str, The name of the entity.It often revolves around a noun-like entity, which can be a person, location, organization, event, or proper noun.",
            "keywords": ["str, retrieval query1 related to the entity", "str, retrieval query2", ...]
        }},
        ...
    ]
}}

## Example

### Example1

**Question**: Which magazine was started first Arthur's Magazine or First for Women?

**Known Knowledge**: None

**Reasoning History**: None

**Hint Entities**: "Arthur's Magazine"

**Output**:

{{
    "entities": [
        {{
            "entity": "Arthur's Magazine",
            "keywords": ["Arthur's Magazine", "Arthur's Magazine start date", "Arthur's Magazine founding date"]
        }},
        {{
            "entity": "First for Women",
            "keywords": ["First for Women", "First for Women start date", "First for Women founding date"]
        }}
    ]
}}

### Example2

**Question**: The Oberoi family is part of a hotel company that has a head office in what city?

**Known Knowledge**: None

**Reasoning History**: None

**Hint Entities**: None

**Output**:

{{
    "entities": [
        {{
            "entity": "Oberoi family",
            "keywords": ["Oberoi family", "Oberoi family hotel company", "Which hotel company is the Oberoi family part of"]
        }}
    ]
}}

### Example3

**Question**: The Oberoi family is part of a hotel company that has a head office in what city?

**Known Knowledge**:

- **Oberoi family**
    - The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.

**Reasoning History**: I have identified that the Oberoi family is part of The Oberoi Group, which is an Indian hotel company. Then I need to find out what city The Oberoi Group is headquartered in.

**Hint Entities**: "Oberoi family", "Indian", "The Oberoi Group"

**Output**:

{{
    "entities": [
        {{
            "entity": "The Oberoi Group",
            "keywords": ["The Oberoi Group", "Where is the head office of The Oberoi Group", "What city is The Oberoi Group headquartered in"]
        }}
    ]
}}
"""

shot_need = """
## Output Format

You should output in JSON format:

{{
    "entities": [
        {{
            "entity": "str, The name of the entity.It often revolves around a noun-like entity, which can be a person, location, organization, event, or proper noun.",
            "keywords": ["str, retrieval query1 related to the entity", "str, retrieval query2", ...]
        }},
        ...
    ]
}}

## Example

### Example1

**Question**: Who, according to articles in Sporting News, stand to make a profit by predicting outcomes such as a team's lead at the end of a quarter or the total points scored, and can also capitalize on event hype, like putting $130 on the Cowboys to potentially gain $100?

**Output**:

{{
    "entities": [
        {{
            "entity": "The person who can make a profit by predicting outcomes",
            "keywords": ["According to articles in Sporting News, who stands to make a profit by predicting outcomes such as a team's lead at the end of a quarter or the total points scored."]
        }},
        {{
            "entity": "The person who can capitalize on event hype to make a profit",
            "keywords": ["According to articles in Sporting News, Who can capitalize on event hype, like putting $130 on the Cowboys to potentially gain $100"]
        }}
    ]
}}

### Example2

**Question**: After the report by Fortune on October 4, 2023, regarding Sam Bankman-Fried's alleged use of Caroline Ellison as a front at Alameda Research, and the subsequent report by TechCrunch involving Sam Bankman-Fried's alleged motives for committing fraud, is the portrayal of Sam Bankman-Fried's actions by both news sources consistent?

**Output**:

{{
    "entities": [
        {{
            "entity": "Sam Bankman-Fried",
            "keywords": [
                "What's the portrayal of Sam Bankman-Fried's actions in the report by Fortune on October 4, 2023, regarding Sam Bankman-Fried's alleged use of Caroline Ellison as a front at Alameda Research?",
                "What's the portrayal of Sam Bankman-Fried's actions in the subsequent report by TechCrunch involving Sam Bankman-Fried's alleged motives for committing fraud?"
            ]
        }}
    ]
}}
"""

prompt_need = """## Background

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
""" + shot_need + """
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

## Output
"""

class SchemaEIItem(BaseModel):
    entity: str
    keywords: list[str]

class SchemaEI(BaseModel):
    entities: list[SchemaEIItem]

response_format_need = {
    "type": "json_schema",
    "json_schema": {
        "name": "EntityKeywords",
        "schema": SchemaEI.model_json_schema()
    }
}


prompt_need_abl2 = """## Background

Currently, there is a knowledge question that needs to be solved, and a retrieval tool can be used to find relevant knowledge to assist you in resolving the issue.

### Task Description

Generate retrieval query based on current reasoning.

## Question currently being solved

### Question

The original question is as follows:

{question}

### Reasoning History

Your previous reasoning history is as follows:

{thought}

## Output

"""

class SchemaEIABL2(BaseModel):
    query: str

response_format_need_abl2 = {
    "type": "json_schema",
    "json_schema": {
        "name": "query",
        "schema": SchemaEIABL2.model_json_schema()
    }
}


# response_format_need = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "EntityKeywords",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "entities": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "entity": {
#                                 "type": "string",
#                                 "description": "The name of the entity.It often revolves around a noun-like entity, which can be a person, location, organization, event, or proper noun."
#                             },
#                             "keywords": {
#                                 "type": "array",
#                                 "items": {
#                                     "type": "string",
#                                     "description": "The retrieval keywords for retrieval related to the entity"
#                                 }
#                             }
#                         },
#                         "required": ["entity", "keywords"],
#                         "additionalProperties": False
#                     }
#                 }
#             },
#             "required": ["entities"],
#             "additionalProperties": False
#         },
#         "strict": True
#     }
# }

shot_learn = """
## Example

### Example1

#### Question

Which magazine was started first Arthur's Magazine or First for Women?

#### Retrieval

Key Entity Retrieved: Arthur's Magazine

Retrieve Queries: When was Arthur's Magazine started?

Retrieved Documents:

##### First Arthur County Courthouse and Jail

The First Arthur County Courthouse and Jail, was perhaps the smallest court house in the United States, and serves now as a museum.

##### Arthur's Magazine

Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into "Godey's Lady's Book"."

#### Output

Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.

### Example2

#### Question

Which magazine was started first Arthur's Magazine or First for Women?

#### Retrieval

Key Entity Retrieved: First for Women

Retrieve Queries: When was First for Women started?

Retrieved Documents:

##### Freeway Complex Fire

The Freeway Complex Fire was a 2008 wildfire in the Santa Ana Canyon area of Orange County, California. The fire started as two separate fires on November 15, 2008. The "Freeway Fire" started first shortly after 9am with the "Landfill Fire" igniting approximately 2 hours later. These two separate fires merged a day later and ultimately destroyed 314 residences in Anaheim Hills and Yorba Linda.

#### Output

None

### Example3

#### Question

Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?

#### Retrieval

Key Entity Retrieved: Move (1970 film)

Retrieve Queries: Who directed the film Move (1970 film)?

Retrieved Documents:

##### Move (1970 film)

Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg.
The screenplay was written by Joel Lieber and Stanley Hart, adapted from a novel by Lieber.

##### Output

Move is a 1970 American comedy film directed by Stuart Rosenberg.
"""

prompt_learn = """## Task Description

You are assisting in solving a QA problem, and you have gathered relevant information using retrieval tools.

Your task is to read and organize the retrieved documents, filtering out irrelevant content while summarizing information pertinent to the current problem. When assessing the usefulness of the content, consider that some information may not appear directly related to the final answer but could be essential for multi-hop reasoning. Even if content does not lead to an immediate conclusion, it may provide necessary context or intermediary insights that help progress toward the answer.

## Note

- Summarize the content directly without adding personal commentary or interpretations. Do not infer or speculate about missing information.
- Preserve the original wording for important content and **ensure that all entity names remain consistent with the original documents**.
""" + shot_learn + """
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

prompt_answer = """### Task Description

You are solving a knowledge-based question, and you have found relevant information using a retrieval tool to help you address the problem.

Now, you need to carefully read the retrieved relevant knowledge and summarize a complete answer based on your previous reasoning.

### Retrieved Relevant Knowledge

{knowledge}

### Question Being Addressed

{question}

### Your Previous Reasoning

{thought}
"""

prompt_infer_abl = """## Background

You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents.

This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts.

## Task Description

Your task is to generate one thought for current step, DON'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'.

### Background Knowledge

{knowledge}

### Question: {question}

Thought: {thought}
"""

# prompt_need_abl = """### Background
#
# Currently, there is a knowledge-based question that needs to be solved, and a retrieval tool can be used to find relevant knowledge to assist in answering the question.
#
# To answer the question, it is essential to gather relevant knowledge, which consists of multiple knowledge segments.
#
# ### Task Description
#
# Your task is to identify the additional knowledge needed based on the current question, existing knowledge, and reasoning history, and generate retrieval queries.
#
# - The generated retrieval keywords will be used by a dense retrieval tool. Ensure that they meet the tool’s requirements to retrieve relevant documents.
# - For the same knowledge point, multiple sub-knowledge queries may be necessary. Ensure that the retrieval queries cover all required aspects.
# - To improve retrieval recall, multiple variations of a query with different expressions may be used, but retain at most two variations for similar meanings.
# - Only generate retrieval queries relevant to the current question and avoid excessive retrieval.
#
# ## Output Format
#
# You should output in JSON format:
#
# {{
#     "entities": ["str, retrieval query1", "str, retrieval query2", ...]
# }}
#
# ## Question currently being solved
#
# ### Known Knowledge
#
# The knowledge that has been retrieved is as follows:
#
# {knowledge}
#
# ### Question
#
# The original question is as follows:
#
# {question}
#
# ### Reasoning History
#
# Your previous reasoning history is as follows:
#
# {thought}
#
# ### Hint Entities
#
# If you need to continue retrieving information on a previously generated knowledge point, ensure that the name of that specific knowledge point remains consistent. The knowledge points that have been retrieved are as follows:
#
# {known_entity}
#
# Use these entities as a reference to identify key knowledge points and generate retrieval keywords accordingly.
#
# If the necessary knowledge points are unclear from the start, you may generate an additional knowledge point named "else" to store retrieval queries for such ambiguous knowledge.
#
# ## Output
# """
#
# response_format_need_abl = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "QueryList",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "queries": {
#                     "type": "array",
#                     "items": {
#                         "type": "string",
#                         "description": "A retrieval query related to the question"
#                     }
#                 }
#             },
#             "required": ["queries"],
#             "additionalProperties": False
#         },
#         "strict": True
#     }
# }
