import numpy as np

def precision_at_n(recommended, relevant, n=10):
    return len(set(recommended[:n]) & set(relevant)) / n

def recall_at_n(recommended, relevant, n=10):
    return len(set(recommended[:n]) & set(relevant)) / len(relevant) if relevant else 0

def hit_ratio_at_n(recommended, relevant, n=10):
    return 1.0 if len(set(recommended[:n]) & set(relevant)) > 0 else 0

def ndcg_at_n(recommended, relevant, n=10):
    dcg = 0.0
    for i, item in enumerate(recommended[:n]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), n)))
    return dcg / idcg if idcg > 0 else 0
