"""Step 4 - robustness and review burden of the Ghost Cluster triage layer (Section 5.6).

For CADEC and PsyTAR: Ward clustering with K fixed by the review-unit size (10-50 expressions per
cluster) on three encoders' embeddings, and HDBSCAN after UMAP over five seeds and on two other encoders.
For each run: Ghost Clusters read, distinct expressions and share of mentions covered, enrichment and
capture of mentions labelled without a concept, label purity of the Ghost Clusters, and the capture
obtained by reading the same number of top-ranked mentions individually. The ranked list is also grouped
by the same clusters ("rank, then group") to count the review units it would form.
Stability: pairwise ARI of partitions and Jaccard overlap of flagged expressions (also for SMM4H).

Outputs (outputs/): results_triage.csv, results_stability.csv
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ghostcluster.clustering import ghost_clusters, hdbscan_after_umap, jaccard, n_clusters, ward_k, ward_labels
from ghostcluster.config import CLUSTER_ENCODER, ENCODERS, OUT_DIR, PRIMARY, WARD_MEAN_SIZE
from ghostcluster.data import load_smm4h
from ghostcluster.scoring import consensus_gap, top_k_similarities

SEEDS = range(5)
MEAN_SIZES = [10, 15, 20, 30, 50]


def load(name):
    return np.load(os.path.join(OUT_DIR, name))


def flagged(texts, labels, gap):
    table = ghost_clusters(labels, gap)
    ghosts = set(table.index[table.ghost])
    return {t for t, c in zip(texts, labels) if c in ghosts}, int(table.ghost.sum())


def triage(run, corpus, texts, labels, gap, sc):
    flag, n_ghost = flagged(texts, labels, gap)
    y = sc.nil.values
    cluster = sc.text.map(dict(zip(texts, labels))).values
    inside = sc.text.isin(flag).values
    top = (sc[PRIMARY].rank(ascending=False, method="first") <= inside.sum()).values
    share = pd.Series(y[inside]).groupby(cluster[inside]).mean()
    majority = np.where(share.reindex(cluster[inside]).values >= 0.5, 1, 0)
    return dict(corpus=corpus, run=run, clusters=n_clusters(labels), ghost_clusters=n_ghost,
                expressions=len(flag), covered=inside.mean(), rate_inside=y[inside].mean(),
                rate_outside=y[~inside].mean(), enrichment=y[inside].mean() / y[~inside].mean(),
                captured=y[inside].sum() / y.sum(), captured_by_ranking=y[top].sum() / y.sum(),
                ghosts_majority_unmappable=float((share >= 0.5).mean()),
                group_decision_agreement=float((majority == y[inside]).mean()),
                ranked_expressions=int(sc.text[top].nunique()),
                ranked_review_units=int(pd.Series(cluster[top]).nunique())), flag


def stability(corpus, runs, flags, names):
    pairs = list(itertools.combinations(names, 2))
    return dict(corpus=corpus, runs=len(names),
                ari=float(np.mean([adjusted_rand_score(runs[a], runs[b]) for a, b in pairs])),
                jaccard=float(np.mean([jaccard(flags[a], flags[b]) for a, b in pairs])),
                jaccard_min=float(np.min([jaccard(flags[a], flags[b]) for a, b in pairs])))


def main():
    rows, stab = [], []
    for corpus in ["cadec", "psytar"]:
        sc = pd.read_csv(os.path.join(OUT_DIR, f"scores_{corpus}.csv"))
        sc["text"] = sc.text.astype(str)
        texts = pd.read_csv(os.path.join(OUT_DIR, f"texts_{corpus}.csv")).text.astype(str).tolist()
        gap = sc.groupby("text")[PRIMARY].mean().reindex(texts).values
        E = load(f"emb_{CLUSTER_ENCODER}_{corpus}.npy")
        runs = {}
        for m in MEAN_SIZES:
            runs[f"Ward, {m} per cluster"] = ward_labels(E, ward_k(len(texts), m))
        for enc in ["sapbert", "biolord"]:
            runs[f"Ward on {enc} embeddings"] = ward_labels(load(f"emb_{enc}_{corpus}.npy"), ward_k(len(texts)))
        for s in SEEDS:
            runs[f"HDBSCAN, seed {s}"] = hdbscan_after_umap(E, s)
        for enc in ["sapbert", "biolord"]:
            runs[f"HDBSCAN on {enc} embeddings"] = hdbscan_after_umap(load(f"emb_{enc}_{corpus}.npy"), 0)
        flags = {}
        for name, lab in runs.items():
            r, flags[name] = triage(name, corpus, texts, lab, gap, sc)
            rows.append(r)
            print(f"{corpus:6s} {name:30s} ghost {r['ghost_clusters']:3d} ({r['expressions']:4d} expr)  covered "
                  f"{r['covered']:.1%}  enrichment {r['enrichment']:.2f}  captured {r['captured']:.1%}  ranking "
                  f"{r['captured_by_ranking']:.1%} in {r['ranked_review_units']} units  majority-unmappable "
                  f"{r['ghosts_majority_unmappable']:.0%}")
        seeds = [n for n in runs if n.startswith("HDBSCAN, seed")]
        wards = [n for n in runs if n.startswith("Ward,")]
        stab.append(stability(corpus, runs, flags, seeds) | dict(method="HDBSCAN across seeds"))
        stab.append(stability(corpus, runs, flags, wards) | dict(method="Ward across review-unit sizes"))
        ref = f"Ward, {WARD_MEAN_SIZE} per cluster"
        assert (ward_labels(E, ward_k(len(texts))) == runs[ref]).all(), "Ward clustering must be deterministic"

    smm = load_smm4h()
    texts = pd.read_csv(os.path.join(OUT_DIR, "texts_smm4h.csv")).text.astype(str).tolist()
    ti = smm.text.map({t: i for i, t in enumerate(texts)}).values
    Eu = load(f"emb_{CLUSTER_ENCODER}_smm4h.npy")
    gaps = []
    for e in ENCODERS:
        top, _ = top_k_similarities(load(f"emb_{e}_smm4h.npy"), load(f"emb_{e}_meddra.npy"), k=5)
        gaps.append(1 - top.mean(1))
    gap = consensus_gap(gaps)[ti]
    runs, flags = {}, {}
    for s in SEEDS:
        runs[s] = hdbscan_after_umap(Eu[ti], s)
        flags[s] = flagged(smm.text.tolist(), runs[s], gap)[0]
    stab.append(stability("smm4h", runs, flags, list(SEEDS)) | dict(method="HDBSCAN across seeds"))
    stab = pd.DataFrame(stab)
    print(stab.to_string(index=False))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "results_triage.csv"), index=False)
    stab.to_csv(os.path.join(OUT_DIR, "results_stability.csv"), index=False)


if __name__ == "__main__":
    main()
