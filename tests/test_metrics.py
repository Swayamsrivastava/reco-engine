import pytest
from eval.metrics import recall_at_k, ndcg_at_k

def test_ndcg_perfect():
    assert ndcg_at_k([1, 2, 3], [1], 3) == pytest.approx(1.0)

def test_ndcg_hit_at_rank_3():                 # 1/log2(4) = 0.5
    assert ndcg_at_k([1, 2, 3], [3], 3) == pytest.approx(0.5)

def test_ndcg_two_hits():                      # (1 + 0.5) / (1 + 1/log2(3))
    assert ndcg_at_k([1, 2, 3], [1, 3], 3) == pytest.approx(0.9197, abs=1e-3)

def test_ndcg_miss():
    assert ndcg_at_k([2, 3, 4], [1], 3) == 0.0

def test_recall():
    assert recall_at_k([1, 2, 3], [1, 9], 3) == 0.5
    assert recall_at_k([1, 2, 3], [2], 1) == 0.0
    assert recall_at_k([1, 2, 3], [], 3) == 0.0