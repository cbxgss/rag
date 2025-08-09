import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle

plt.rcParams['font.family'] = 'Times New Roman'


def get_step_retrive_nums(method: str, log: dict):
    score = log["evaluate"][0]
    trace = log["trace"]
    if method == "hrag":
        r_nums = 0
        step_nums = 0
        for step_i, trace_step in trace["trace"].items():
            if "retrive" in trace_step:
                step_nums += 1
                for entity, v in trace_step["retrive"].items():
                    r_nums += len(v["r"])
        return score, max(1, step_nums), r_nums
    elif method == "ircot":
        return score, len(trace) + 1, len(trace)
    elif method == "genground":
        r_nums = 0
        for step_i, trace_step in trace["history"].items():
            r_nums += len(trace_step)
        return score, len(trace["history"]), r_nums
    elif method == "metarag":
        return score, len(trace["interaction_log"]) + 1, len(trace["interaction_log"]) + 1
    else:
        raise ValueError(f"Unknown method: {method}")

def extract_log_dir(method: str, dir: str):
    log_dir = f"{dir}/output"
    json_path = [os.path.join(log_dir, p) for p in os.listdir(log_dir) if p.endswith(".json")]
    csv_content = []
    for q_id, path in enumerate(tqdm(json_path, desc=f"Extracting logs from {dir}")):
        with open(path, "r") as f:
            log = json.load(f)
            score, step_nums, retrive_nums = get_step_retrive_nums(method, log)
            csv_content.append([q_id, score, step_nums, retrive_nums])
    return csv_content

def extract_log():
    log_dirs = {
        "hotpotqa": {
            "ircot": ["72b-1000/0107_131731-ircot"],
            "genground": ["72b-1000/0121_092159-genground"],
            "metarag": ["72b-1000/0119_081423-metarag"],
            "hrag": [
                "72b-1000/0107_131526-hrag",
                "abl-sft/0204_013836-hrag",
                "abl-sft/wo1-0130_125133-hrag",
                "abl-sft/wo2-0130_160244-hrag",
                "abl-sft/wo3-0130_160457-hrag",
            ],
        },
        "2wikimultihopqa": {
            "ircot": ["72b-1000/0107_051557-ircot"],
            "genground": ["72b-1000/0121_092554-genground"],
            "metarag": ["72b-1000/0118_090605-metarag"],
            "hrag": [
                "72b-1000/0107_022909-hrag",
                "abl-sft/0204_013915-hrag",
                "abl-sft/wo1-0130_124000-hrag",
                "abl-sft/wo2-0130_160302-hrag",
                "abl-sft/wo3-0130_160438-hrag",
            ],
        },
        "musique": {
            "ircot": ["72b-1000/0107_065547-ircot"],
            "genground": ["72b-1000/0121_092848-genground"],
            "metarag": ["72b-1000/0119_081403-metarag"],
            "hrag": [
                "72b-1000/0107_065430-hrag",
                "abl-sft/0204_013952-hrag",
                "abl-sft/wo1-0130_123927-hrag",
                "abl-sft/wo2-0130_160323-hrag",
                "abl-sft/wo3-0130_160350-hrag",
            ]
        },
    }

    out = {}
    for dataset, dataset_dirs in log_dirs.items():
        out[dataset] = {}
        for method in dataset_dirs:
            for dir in dataset_dirs[method]:
                out[dataset][dir] = extract_log_dir(method, f"log/rag/{dataset}/{dir}")

    os.makedirs("tmp/exp", exist_ok=True)
    with open("tmp/exp/exp.json", "w") as f:
        json.dump(out, f)

def draw():
    import seaborn as sns
    log_dirs = {
        "hotpotqa": {
            "step": {
                "72b-1000/0107_131526-hrag": "DualRAG",
                "72b-1000/0107_131731-ircot": "IRCoT",
                # "72b-1000/0121_092159-genground": "GenGround",
                "72b-1000/0119_081423-metarag": "MetaRAG",
            },
            "retrive": {
                "abl-sft/0204_013836-hrag": "DualRAG-FT",
                "abl-sft/wo2-0130_160244-hrag": "DualRAG",
            },
        },
        "2wikimultihopqa": {
            "step": {
                "72b-1000/0107_022909-hrag": "DualRAG",
                "72b-1000/0107_051557-ircot": "IRCoT",
                # "72b-1000/0121_092554-genground": "GenGround",
                "72b-1000/0118_090605-metarag": "MetaRAG",
            },
            "retrive": {
                "abl-sft/0204_013915-hrag": "DualRAG-FT",
                "abl-sft/wo2-0130_160302-hrag": "DualRAG",
            },
        },
        "musique": {
            "step": {
                "72b-1000/0107_065430-hrag": "DualRAG",
                "72b-1000/0107_065547-ircot": "IRCoT",
                # "72b-1000/0121_092848-genground": "GenGround",
                "72b-1000/0119_081403-metarag": "MetaRAG",
            },
            "retrive": {
                "abl-sft/0204_013952-hrag": "DualRAG-FT",
                "abl-sft/wo2-0130_160323-hrag": "DualRAG",
            }
        },
    }
    target_order_steps = ['DualRAG', 'IRCoT', 'MetaRAG']
    target_order_r = ['DualRAG-FT', 'DualRAG']

    with open("tmp/exp/exp.json", "r") as f:
        data = json.load(f)
    # method_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 蓝色、橙色、绿色
    # dataset_bg_colors = ["#e8f4fa", "#f0f0f0", "#f4e8fa"]  # 淡蓝、淡灰、淡紫
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), dpi=300)
    # for i, (ax, (dataset, dataset_data), bg_color) in enumerate(zip(axes, data.items(), dataset_bg_colors)):
    #     boxplot_data = []
    #     labels = []
    #     for dir, dir_data in dataset_data.items():
    #         if dir not in log_dirs[dataset]["step"]:
    #             continue
    #         method = log_dirs[dataset]["step"][dir]
    #         step_nums = [min(d[2], 5) for d in dir_data if d[1] > 0]
    #         if step_nums:
    #             boxplot_data.append(step_nums)
    #             labels.append(method)
    #     sorted_indices = [labels.index(method) for method in target_order_steps if method in labels]
    #     boxplot_data = [boxplot_data[j] for j in sorted_indices]
    #     labels = [labels[j] for j in sorted_indices]
    #     bp = ax.boxplot(boxplot_data, labels=labels, vert=True, patch_artist=True, notch=False,
    #                     boxprops=dict(linewidth=1, color='black'),
    #                     whiskerprops=dict(linewidth=1, linestyle="--", color='black'),
    #                     capprops=dict(linewidth=1, color='black'),
    #                     medianprops=dict(linewidth=2.5, color='black'),
    #                     meanprops=dict(marker='o', markerfacecolor='black', markersize=5),
    #                     flierprops=dict(marker='o', markersize=4, markerfacecolor='gray', alpha=0.6),
    #                     showmeans=True)
    #     for patch, color in zip(bp["boxes"], method_colors[:len(boxplot_data)]):
    #         patch.set_facecolor(color)
    #         patch.set_alpha(0.7)
    #     ax.set_facecolor(bg_color)  # 设置背景颜色
    #     ax.tick_params(axis="x", labelsize=20, rotation=0)
    #     ax.tick_params(axis="y", labelsize=20)
    #     if i == 0:
    #         ax.set_ylabel("Step Numbers", fontsize=24)
    #     if i == 1:
    #         ax.set_xlabel("Methods", fontsize=24)
    #     ax.set_title(f"{dataset}", fontsize=24)
    #     ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.set_yticks(range(0, 6))  # 只显示整数刻度
    # plt.tight_layout()
    # plt.savefig("tmp/exp/steps_box.pdf", dpi=300, bbox_inches="tight", format="pdf")

    method_colors = {"DualRAG": "#1f77b4", "IRCoT": "#ff7f0e", "MetaRAG": "#2ca02c"}  # 方法对应颜色
    dataset_names = list(data.keys())  # 获取数据集名称
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    boxplot_data = []
    boxplot_positions = []
    colors = []
    labels = []
    method_order = ["DualRAG", "IRCoT", "MetaRAG"]
    x_positions = {dataset: i + 1 for i, dataset in enumerate(dataset_names)}  # x 轴位置映射
    for dataset_i, dataset in enumerate(dataset_names):
        ax.add_patch(Rectangle((dataset_i + 0.55, 0), 0.9, 5.2, facecolor="#e8e8e8", alpha=0.5, edgecolor="black", linewidth=1.5))  # 灰色背景框
        dataset_data = data[dataset]
        for method in method_order:
            method_data = []
            for dir, dir_data in dataset_data.items():
                if dir not in log_dirs[dataset]["step"]:
                    continue
                if log_dirs[dataset]["step"][dir] == method:
                    step_nums = [min(d[2], 5) for d in dir_data if d[1] > 0]
                    if step_nums:
                        method_data.extend(step_nums)
            if method_data:
                boxplot_data.append(method_data)
                boxplot_positions.append(x_positions[dataset] + (method_order.index(method) - 1) * 0.2)  # 偏移位置
                colors.append(method_colors[method])
                labels.append(method)
                print(method, dataset, len(method_data), sum(method_data) / len(method_data))
    exit()
    bp = ax.boxplot(boxplot_data, positions=boxplot_positions, vert=True, patch_artist=True, notch=True, widths=0.15,
                    boxprops=dict(linewidth=1, color='black'),
                    whiskerprops=dict(linewidth=1, linestyle="--", color='black'),
                    capprops=dict(linewidth=1, color='black'),
                    medianprops=dict(linewidth=2.5, color='black'),
                    meanprops=dict(marker='o', markerfacecolor='black', markersize=5),
                    flierprops=dict(marker='o', markersize=4, markerfacecolor='gray', alpha=0.6),
                    showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(list(x_positions.values()))  # 只在数据集的位置放置刻度
    ax.set_xticklabels(dataset_names, fontsize=22)
    ax.set_yticks(range(0, 6))
    ax.tick_params(axis="y", labelsize=22)
    ax.set_ylabel("Step Numbers", fontsize=22)
    # ax.set_title("Comparison of Methods Across Datasets", fontsize=18)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_patches = [Patch(color=method_colors[m], label=m) for m in method_order]
    ax.legend(handles=legend_patches, title="Methods", fontsize=12, loc="best")
    plt.tight_layout()
    plt.savefig("tmp/exp/steps_box.pdf", dpi=300, bbox_inches="tight", format="pdf")

    # 图2: r_nums~ 分布图
    sns.set_theme(style="whitegrid", font_scale=1.5)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(8, 6))
    merged_r_nums = {}
    for dataset, dataset_data in data.items():
        for dir, dir_data in dataset_data.items():
            if dir not in log_dirs[dataset]["retrive"]:
                continue
            method = log_dirs[dataset]["retrive"][dir]
            r_nums = [d[3] for d in dir_data if d[1] > 0]
            # 合并数据
            if method not in merged_r_nums:
                merged_r_nums[method] = []
            merged_r_nums[method].extend(r_nums)
    colors = sns.color_palette("muted", len(merged_r_nums))
    for i, (method, r_nums) in enumerate(merged_r_nums.items()):
        sns.histplot(r_nums, bins=50, kde=True, label=method, element="step", stat="density",
                     color=colors[i], alpha=0.6, linewidth=2)
    plt.xlabel("The Count of Generated Queries", fontsize=22)
    plt.ylabel("Distribution for the Queries Count", fontsize=22)
    plt.xlim(0, 21)
    plt.xticks(range(0, 21), fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tmp/exp/r_sns.pdf", dpi=300, bbox_inches='tight', format="pdf")  # 高分辨率保存
    plt.close()

# extract_log()
draw()
