"""Step 4 - robustness of the Ghost Cluster triage layer (Section 5.5).

For CADEC and PsyTAR: HDBSCAN after UMAP over five seeds and on two other encoders' embeddings, and
deterministic Ward clustering with K equal to the HDBSCAN cluster count (and K +/- 20%). For each run:
Ghost Clusters read, share of expressions covered, enrichment and capture of human-judged unmappable
mentions, and the capture obtained by reading the same number of top-ranked expressions individually.
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
from ghostcluster.clustering import ghost_clusters, hdbscan_after_umap, jaccard, n_clusters, ward_labels
from ghostcluster.config import CLUSTER_ENCODER, OUT_DIR
from ghostcluster.data import load_smm4h
from ghostcluster.scoring import semantic_gap

SEMANTIC = f"{CLUSTER_ENCODER}|top5|PT+LLT"
SEEDS = range(5)


def load(name):
    return np.load(os.path.join(OUT_DIR, name))


def flagged(texts, labels, gap):
    table = ghost_clusters(labels, gap)
    ghosts = set(table.index[table.ghost])
    return {t for t, c in zip(texts, labels) if c in ghosts}, int(table.ghost.sum())


def triage(run, corpus, texts, labels, gap, sc):
    flag, n_ghost = flagged(texts, labels, gap)
    y = sc.nil.values
    inside = sc.text.isin(flag).values
    top = (sc[SEMANTIC].rank(ascending=False, method="first") <= inside.sum()).values
    return dict(corpus=corpus, run=run, clusters=n_clusters(labels), ghost_clusters=n_ghost,
                covered=inside.mean(), rate_inside=y[inside].mean(), rate_outside=y[~inside].mean(),
                enrichment=y[inside].mean() / y[~inside].mean(), captured=y[inside].sum() / y.sum(),
                captured_by_ranking=y[top].sum() / y.sum()), flag


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
        gap = sc.groupby("text")[SEMANTIC].mean().reindex(texts).values
        E = load(f"emb_{CLUSTER_ENCODER}_{corpus}.npy")
        runs = {f"HDBSCAN, seed {s}": hdbscan_after_umap(E, s) for s in SEEDS}
        for enc in ["sapbert", "biolord"]:
            runs[f"HDBSCAN on {enc} embeddings"] = hdbscan_after_umap(load(f"emb_{enc}_{corpus}.npy"), 0)
        k = n_clusters(runs["HDBSCAN, seed 0"])
        ward = {"K": k, "0.8K": round(0.8 * k), "1.2K": round(1.2 * k)}
        for name, kk in ward.items():
            runs[f"Ward, {name} = {kk}"] = ward_labels(E, kk)
        flags = {}
        for name, lab in runs.items():
            r, flags[name] = triage(name, corpus, texts, lab, gap, sc)
            rows.append(r)
            print(f"{corpus:6s} {name:30s} ghost {r['ghost_clusters']:3d}  covered {r['covered']:.1%}  "
                  f"enrichment {r['enrichment']:.2f}  captured {r['captured']:.1%}  ranking {r['captured_by_ranking']:.1%}")
        seeds = [n for n in runs if n.startswith("HDBSCAN, seed")]
        wards = [n for n in runs if n.startswith("Ward")]
        stab.append(stability(corpus, runs, flags, seeds) | dict(method="HDBSCAN across seeds"))
        stab.append(stability(corpus, runs, flags, wards) | dict(method="Ward across K"))
        assert (ward_labels(E, k) == runs[wards[0]]).all(), "Ward clustering must be deterministic"

    smm = load_smm4h()
    texts = pd.read_csv(os.path.join(OUT_DIR, "texts_smm4h.csv")).text.astype(str).tolist()
    ti = smm.text.map({t: i for i, t in enumerate(texts)}).values
    Eu = load(f"emb_{CLUSTER_ENCODER}_smm4h.npy")
    gap = semantic_gap(Eu, load(f"emb_{CLUSTER_ENCODER}_meddra.npy"))[ti]
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
