import os
from tqdm import tqdm
import tiktoken


tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")


def cal_str(s: str) -> int:
    input_ids = tokenizer.encode(s)
    return len(input_ids)

def cal(log: str):
    token = 0
    for file in os.listdir(f"{log}/llm"):
        if file.endswith("_rsp.md") and "answer_rsp.md" not in file and "reanswer_rsp.md" not in file and "evaluate_rsp.md" not in file:
            with open(f"{log}/llm/{file}", "r") as f:
                s = f.read()
                token += cal_str(s)
    return token


def cal_dir(path: str):
    token = 0
    for file in tqdm(os.listdir(f"{path}/log"), leave=False):
        if os.path.isdir(f"{path}/log/{file}"):
            token += cal(f"{path}/log/{file}")
    return token


nums = {
    "direct": "/aigc_sgply_ssd/dujizhen03/chengrong/workspace/rag/log/rag/2wikimultihopqa/72b-1000/0107_061742-direct",
    "native": "/aigc_sgply_ssd/dujizhen03/chengrong/workspace/rag/log/rag/2wikimultihopqa/72b-1000/0107_061350-native",
    "ircot": "/aigc_sgply_ssd/dujizhen03/chengrong/workspace/rag/log/rag/2wikimultihopqa/72b-1000/0107_051557-ircot",
    # "metarag": "/aigc_sgply_ssd/dujizhen03/chengrong/workspace/rag/log/rag/2wikimultihopqa/72b-1000/0118_090605-metarag",
    "genground": "/aigc_sgply_ssd/dujizhen03/chengrong/workspace/rag/log/rag/2wikimultihopqa/72b-1000/0121_092554-genground",
    "dualrag": "/aigc_sgply_ssd/dujizhen03/chengrong/workspace/rag/log/rag/2wikimultihopqa/72b-1000/0107_022909-hrag",
}


for k, v in tqdm(nums.items()):
    print(k, cal_dir(v))
