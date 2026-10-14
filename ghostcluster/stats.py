"""Evaluation metrics with document-level (cluster) bootstrap confidence intervals."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import N_BOOTSTRAP


def _resample(groups, rng):
    uniq = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in uniq}
    while True:
        yield np.concatenate([index[g] for g in rng.choice(uniq, len(uniq))])


def bootstrap_metrics(y, s, groups, rng, n=N_BOOTSTRAP):
    """AUROC and AUPRC with 95% percentile intervals, resampling documents (posts or reviews)."""
    y, s = np.asarray(y), np.asarray(s)
    auc, ap, a, p = roc_auc_score(y, s), average_precision_score(y, s), [], []
    draws = _resample(groups, rng)
    for _ in range(n):
        ix = next(draws)
        if y[ix].min() == y[ix].max():
            continue
        a.append(roc_auc_score(y[ix], s[ix]))
        p.append(average_precision_score(y[ix], s[ix]))
    return dict(auroc=auc, auroc_lo=np.percentile(a, 2.5), auroc_hi=np.percentile(a, 97.5),
                auprc=ap, auprc_lo=np.percentile(p, 2.5), auprc_hi=np.percentile(p, 97.5))


def paired_auprc_difference(y, s1, s2, groups, rng, n=N_BOOTSTRAP):
    """AUPRC(s1) - AUPRC(s2) on the full data, with 95% interval and two-sided p-value from paired document resamples."""
    y = np.asarray(y)
    d, draws = [], _resample(groups, rng)
    for _ in range(n):
        ix = next(draws)
        if y[ix].min() == y[ix].max():
            continue
        d.append(average_precision_score(y[ix], s1[ix]) - average_precision_score(y[ix], s2[ix]))
    d = np.asarray(d)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / len(d))
    point = average_precision_score(y, s1) - average_precision_score(y, s2)
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)
