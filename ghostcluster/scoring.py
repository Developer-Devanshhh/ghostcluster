"""Gap scores: semantic distance and lexical distance to the terminology."""
import numpy as np
from rapidfuzz import fuzz, process

from .data import norm


def top_k_similarities(Q, M, k=5, batch=512):
    """Top-k cosine similarities (descending) of each row of Q against M; both L2-normalised."""
    top = np.zeros((len(Q), k), dtype=np.float32)
    arg = np.zeros((len(Q), k), dtype=np.int64)
    for i in range(0, len(Q), batch):
        S = Q[i:i + batch] @ M.T
        a = np.argpartition(-S, k, axis=1)[:, :k]
        v = np.take_along_axis(S, a, 1)
        o = np.argsort(-v, axis=1)
        top[i:i + batch] = np.take_along_axis(v, o, 1)
        arg[i:i + batch] = np.take_along_axis(a, o, 1)
    return top, arg


def semantic_gap(Q, M, k=5):
    """g_i = 1 - mean of the k largest cosine similarities (Eq. 1); k=1 gives the max-similarity variant."""
    top, _ = top_k_similarities(Q, M, k=k)
    return 1.0 - top[:, :k].mean(1)


def lexical_gap(texts, term_names, batch=200):
    """l_i = 1 - best token-sort ratio (0-100 scaled to 0-1) between the text and any term name."""
    choices = list(dict.fromkeys(norm(t) for t in term_names))
    best = np.zeros(len(texts))
    for i in range(0, len(texts), batch):
        sims = process.cdist([norm(t) for t in texts[i:i + batch]], choices,
                             scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.uint8)
        best[i:i + batch] = sims.max(1)
    return 1.0 - best / 100.0


def word_count(texts):
    return np.array([len(str(t).split()) for t in texts], dtype=float)
