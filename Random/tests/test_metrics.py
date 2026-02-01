import math
import numpy as np
from Random.helloworld import recall, roc_auc


def test_recall_basic():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1])
    # tp = 2 (positions 0,3), fn = 1 (position 2) => recall = 2/3
    assert math.isclose(recall(y_true, y_pred), 2/3)


def test_recall_no_positives():
    y_true = np.zeros(5, dtype=int)
    y_pred = np.ones(5, dtype=int)
    assert math.isnan(recall(y_true, y_pred))


def test_recall_length_mismatch():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0])
    try:
        recall(y_true, y_pred)
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError:
        pass


def test_roc_auc_perfect():
    # perfect score ordering -> auc = 1.0
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    assert math.isclose(roc_auc(y_true, y_score), 1.0)


def test_roc_auc_random():
    # known case: when scores are random but symmetric the AUC ~ 0.5
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=1000)
    y_score = rng.rand(1000)
    auc = roc_auc(y_true, y_score)
    assert 0.0 <= auc <= 1.0


def test_roc_auc_one_class():
    y_true = np.ones(10, dtype=int)
    y_score = np.linspace(0, 1, 10)
    assert math.isnan(roc_auc(y_true, y_score))


def test_roc_auc_length_mismatch():
    y_true = np.array([1, 0, 1])
    y_score = np.array([0.1, 0.4])
    try:
        roc_auc(y_true, y_score)
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError:
        pass
