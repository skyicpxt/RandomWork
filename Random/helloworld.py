import numpy as np
from typing import Union


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall (true positive rate) for binary labels 0/1.

    Returns NaN when there are no positive samples in y_true.
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    denom = tp + fn
    return float(tp) / denom if denom else float('nan')


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC for binary labels using the trapezoidal rule.

    y_true must be 0/1 labels. y_score are continuous scores where larger
    means more likely positive.
    Returns NaN if y_true contains only one class.
    """
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return float('nan')

    # sort by descending score
    desc_idx = np.argsort(-y_score)
    y_true_sorted = y_true[desc_idx]

    tps = np.cumsum(y_true_sorted == 1).astype(float)
    fps = np.cumsum(y_true_sorted == 0).astype(float)

    tpr = np.concatenate(([0.0], tps / positives))
    fpr = np.concatenate(([0.0], fps / negatives))

    auc = np.trapz(tpr, fpr)
    return float(auc)


if __name__ == "__main__":
    # small example run when executed directly
    y_true = np.random.randint(0, 2, 20)
    y_scores = np.random.rand(20)
    y_pred = (y_scores >= 0.5).astype(int)

    print("recall:", recall(y_true=y_true, y_pred=y_pred))
    print("roc_auc:", roc_auc(y_true=y_true, y_score=y_scores))
