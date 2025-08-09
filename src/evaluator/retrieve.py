import pytrec_eval


def retrieval_metrics(scores, relevance, k_values=None):
    """
    scores = {
        'q1': {'d10': 2.5, 'd2': 1.8, 'd15': 1.2},
        'q2': {'d5': 3.0, 'd8': 2.2, 'd12': 1.5},
        'q3': {'d3': 2.8, 'd7': 2.0, 'd11': 1.6}
    }
    # ground truth
    relevance = {
        'q1': {'d10': 1, 'd2': 0, 'd15': 0},
        'q2': {'d5': 1, 'd8': 0, 'd12': 1},
        'q3': {'d3': 1, 'd7': 0, 'd11': 0}
    }
    """
    k_values = [1, 5, 10, 20, 50, 100] if k_values is None else k_values

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        relevance,
        {map_string, ndcg_string, recall_string, precision_string}
    )
    results = evaluator.evaluate(scores)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"ndcg@{k}"] = 0.0
        _map[f"map@{k}"] = 0.0
        recall[f"recall@{k}"] = 0.0
        precision[f"precision@{k}"] = 0.0

    for query_id in results.keys():
        for k in k_values:
            ndcg[f"ndcg@{k}"] += results[query_id]["ndcg_cut_" + str(k)]
            _map[f"map@{k}"] += results[query_id]["map_cut_" + str(k)]
            recall[f"recall@{k}"] += results[query_id]["recall_" + str(k)]
            precision[f"precision@{k}"] += results[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"ndcg@{k}"] = round(ndcg[f"ndcg@{k}"] / len(results), 5)
        _map[f"map@{k}"] = round(_map[f"map@{k}"] / len(results), 5)
        recall[f"recall@{k}"] = round(recall[f"recall@{k}"] / len(results), 5)
        precision[f"precision@{k}"] = round(precision[f"precision@{k}"] / len(results), 5)

    return {
        "recall": recall,
        "precision": precision,
        "ndcg": ndcg,
        "map": _map,
    }


if __name__ == "__main__":
    import json
    # 分数具体数值不重要，只要排序一样，结果就一样
    scores = {
        'q1': {'d10': 2.5, 'd2': 1.8, 'd15': 1.2},
        'q2': {'d5': 3.0, 'd8': 2.2, 'd12': 1.5},
        'q3': {'d3': 2.8, 'd7': 2.0, 'd11': 1.6}
    }
    # 以下2个结果一样
    scores = {
        'q1': {'d10': 1, 'd2': 0, 'd15': 0},
        'q2': {'d5': 1, 'd8': 0, 'd12': 1},
        'q3': {'d3': 1, 'd7': 0, 'd11': 0}
    }
    scores = {
        'q1': {'d10': 1},
        'q2': {'d5': 1,'d12': 1},
        'q3': {'d3': 1}
    }
    relevance = {
        'q1': {'d10': 1, 'd2': 0, 'd15': 0},
        'q2': {'d5': 1, 'd8': 0, 'd12': 1},
        'q3': {'d3': 1, 'd7': 0, 'd11': 0}
    }
    metrics = retrieval_metrics(scores, relevance)
    print(json.dumps(metrics, indent=4))
