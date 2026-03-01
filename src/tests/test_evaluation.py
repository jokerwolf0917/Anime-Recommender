
import numpy as np
from evaluation import precision_at_n, recall_at_n, hit_ratio_at_n, ndcg_at_n

def test_metrics_small():
    recommended = [1,2,3,4,5]
    relevant = [3,5,7]
    assert abs(precision_at_n(recommended, relevant, n=5) - 2/5) < 1e-9
    assert abs(recall_at_n(recommended, relevant, n=5) - 2/3) < 1e-9
    assert hit_ratio_at_n(recommended, relevant, n=5) == 1.0
    v = ndcg_at_n(recommended, relevant, n=5)
    assert 0.0 <= v <= 1.0
