"""Clustering procedures and Ghost Cluster selection."""
import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering

from .config import GHOST_QUANTILE, HDBSCAN_PARAMS, MIN_CLUSTER_SIZE_GHOST, UMAP_PARAMS, WARD_MEAN_SIZE


def umap_project(E, seed):
    return umap.UMAP(random_state=seed, **UMAP_PARAMS).fit_transform(E)


def hdbscan_labels(X, **params):
    return hdbscan.HDBSCAN(metric="euclidean", **{**HDBSCAN_PARAMS, **params}).fit_predict(X)


def hdbscan_after_umap(E, seed):
    """UMAP to five dimensions, then HDBSCAN. Noise points get label -1."""
    return hdbscan_labels(umap_project(E, seed))


def ward_k(n_expressions, mean_size=WARD_MEAN_SIZE):
    """Number of Ward clusters for a corpus: fixed in advance by the intended size of a review unit."""
    return max(2, int(round(n_expressions / mean_size)))


def ward_labels(E, k):
    """Deterministic agglomerative clustering with Ward linkage on L2-normalised embeddings."""
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    return AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit_predict(En)


def dbcv(X, labels):
    """Density-Based Clustering Validation index (Moulavi et al., 2014)."""
    try:
        return float(hdbscan.validity.validity_index(X.astype(np.float64), labels))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def n_clusters(labels):
    return len(set(labels) - {-1})


def ghost_clusters(labels, gap_scores):
    """Return (table of eligible clusters with mean gap score, boolean Ghost flag) for one partition.

    Eligible clusters have at least MIN_CLUSTER_SIZE_GHOST members; Ghost Clusters are those whose
    mean gap score lies above the GHOST_QUANTILE of eligible clusters.
    """
    d = pd.DataFrame({"cluster": labels, "gap": gap_scores})
    d = d[d.cluster >= 0]
    size = d.groupby("cluster").size()
    eligible = size.index[size >= MIN_CLUSTER_SIZE_GHOST]
    table = d[d.cluster.isin(eligible)].groupby("cluster").gap.agg(["mean", "size"])
    table["ghost"] = table["mean"] > table["mean"].quantile(GHOST_QUANTILE)
    return table


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)
