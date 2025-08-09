import sys
sys.path.append(".")
import re
import pandas as pd
from tabulate import tabulate

from scripts.logs import logs


dataset = "2wikimultihopqa" # "nq", "eli5", "asqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle", "strategyqa"
print(f"dataset: {dataset}")


def read_log(log_dir, column="llm_judge"):
    csv_path = f"log/rag/{dataset}/{log_dir}/evaluate.csv"
    csv = pd.read_csv(csv_path, index_col="id")
    # 按 llm_judge，得出 right 和 error 的 id
    right_id = csv[csv[column] == 1].index.tolist()
    error_id = csv[csv[column] == 0].index.tolist()
    # 提取成 id 中数字
    right_id = [int(re.search(r"\d+", id).group()) for id in right_id]
    error_id = [int(re.search(r"\d+", id).group()) for id in error_id]
    acc = csv.loc["mean", column]
    return right_id, error_id, acc


def read_log_n(log_dirs, column="llm_judge"):
    # right_id 为全部 log 的 right_id 的交集
    # mid_id 为摇摆题目，即在某些 log 中为 right，某些 log 中为 error
    # error_id 为全部 log 的 error_id 的交集
    right_id_li = []
    error_id_li = []
    acc_li = []
    for log_dir in log_dirs:
        right_id, error_id, acc = read_log(log_dir, column)
        right_id_li.append(right_id)
        error_id_li.append(error_id)
        acc_li.append(acc)
    right_id = set(right_id_li[0])
    error_id = set(error_id_li[0])
    for i in range(1, len(right_id_li)):
        right_id = right_id.intersection(right_id_li[i])
        error_id = error_id.intersection(error_id_li[i])
    mid_id = set(right_id_li[0] + error_id_li[0]) - right_id - error_id
    # 返回排序后的list
    right_id = sorted(list(right_id))
    mid_id = sorted(list(mid_id))
    error_id = sorted(list(error_id))
    return right_id, mid_id, error_id, sum(acc_li) / len(acc_li), len(acc_li)


def read_metrics(log_dir) -> pd.Series:
    csv_path = f"log/rag/{dataset}/{log_dir}/evaluate.csv"
    csv = pd.read_csv(csv_path, index_col="id")
    mean = csv.loc["mean"]
    return mean


def read_metrics_n(log_dirs):
    metrics_li = []
    for log_dir in log_dirs:
        metrics = read_metrics(log_dir)
        metrics_li.append(metrics)
    metrics_df = pd.concat(metrics_li, axis=1)
    netrics_keys = ["llm_judge"] + [m for m in metrics_df.index if m != "llm_judge"]
    metrics_df = metrics_df.loc[netrics_keys]
    metrics_mean = metrics_df.mean(axis=1)
    return metrics_mean


log_dict: dict = logs[dataset]
metrics = []
re_id = []
for log in log_dict:
    metrics.append(read_metrics_n(log_dict[log]))
    right_id, mid_id, error_id, acc, num = read_log_n(log_dict[log], column="llm_judge")
    re_id.append([num, acc, f"✅{len(right_id)}, ❌{len(error_id)}, {len(mid_id)}", mid_id])

re_id = pd.DataFrame(re_id, index=log_dict.keys(), columns=["num", "llm_judge", "len", "mid_id"]).round(3)
print(tabulate(re_id, headers='keys', tablefmt='pretty', stralign='left'))
metrics = pd.DataFrame(metrics, index=log_dict.keys()).round(3)
print(tabulate(metrics, headers='keys', tablefmt='pretty', stralign='left'))


# import os
# for dataset in logs:
#     for log in logs[dataset]:
#         for log_dir in logs[dataset][log]:
#             log_dir_new = log_dir.replace("-seed0", "")
#             os.rename(f"log/rag/{dataset}/{log_dir}", f"log/rag/{dataset}/{log_dir_new}")
#             print(f"rename {log_dir} to {log_dir_new}")
