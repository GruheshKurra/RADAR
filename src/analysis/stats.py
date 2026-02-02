import numpy as np
from scipy import stats
from scipy.stats import norm
from typing import Tuple


def bootstrap_auc_ci(labels: np.ndarray, probs: np.ndarray,
                     n_bootstrap: int = 2000, confidence: float = 0.95) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(42)
    aucs = []
    n = len(labels)

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        aucs.append(roc_auc_score(labels[indices], probs[indices]))

    aucs = np.array(aucs)
    alpha = (1 - confidence) / 2

    return (
        np.percentile(aucs, alpha * 100),
        np.percentile(aucs, (1 - alpha) * 100),
    )


def delong_test(labels: np.ndarray, probs1: np.ndarray, probs2: np.ndarray) -> Tuple[float, float]:
    n1 = np.sum(labels == 1)
    n0 = np.sum(labels == 0)

    pos_probs1 = probs1[labels == 1]
    neg_probs1 = probs1[labels == 0]
    pos_probs2 = probs2[labels == 1]
    neg_probs2 = probs2[labels == 0]

    def placement_values(pos, neg):
        V10 = np.array([np.sum(pos > n) + 0.5 * np.sum(pos == n) for n in neg]) / len(pos)
        V01 = np.array([np.sum(p > neg).sum() + 0.5 * np.sum(p == neg).sum() for p in pos]) / len(neg)
        return V01, V10

    pv1_pos, pv1_neg = placement_values(pos_probs1, neg_probs1)
    pv2_pos, pv2_neg = placement_values(pos_probs2, neg_probs2)

    auc1 = np.mean(pv1_pos)
    auc2 = np.mean(pv2_pos)

    v1_pos = np.var(pv1_pos, ddof=1)
    v1_neg = np.var(pv1_neg, ddof=1)
    v2_pos = np.var(pv2_pos, ddof=1)
    v2_neg = np.var(pv2_neg, ddof=1)

    cov_pos = np.cov(pv1_pos, pv2_pos)[0, 1] if len(pv1_pos) > 1 else 0
    cov_neg = np.cov(pv1_neg, pv2_neg)[0, 1] if len(pv1_neg) > 1 else 0

    var_auc1 = v1_pos / n1 + v1_neg / n0
    var_auc2 = v2_pos / n1 + v2_neg / n0
    cov_auc = cov_pos / n1 + cov_neg / n0

    var_diff = var_auc1 + var_auc2 - 2 * cov_auc

    if var_diff <= 0:
        return 0.0, 1.0

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return z, p_value
