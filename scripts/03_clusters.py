"""Step 3 - clustering configuration, Ghost Cluster content and cluster-level scoring (Sections 3.6, 5.6).

A. SMM4H-2017: silhouette and DBCV for the 24-configuration HDBSCAN grid; most distant Ghost Clusters.
B. CADEC and PsyTAR: Ward clustering (primary, K fixed by the review-unit size) and HDBSCAN after UMAP.
   For every Ghost Cluster: size, distinct expressions, share of mentions labelled without a concept,
   concept coherence (share of coded members sharing the most frequent gold concept), and the nearest
   MedDRA terms to the cluster centroid (for manual inspection of the kind of gap).
   Cluster-averaged scores are compared with expression-level scores.

Outputs (outputs/): smm4h_grid.csv, ghost_clusters_<corpus>_<method>.csv, results_cluster_scoring.csv
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ghostcluster.clustering import (dbcv, ghost_clusters, hdbscan_after_umap, hdbscan_labels, n_clusters,
                                     umap_project, ward_k, ward_labels)
from ghostcluster.config import CLUSTER_ENCODER, ENCODERS, HDBSCAN_GRID, HDBSCAN_PARAMS, OUT_DIR, PRIMARY, SEED
from ghostcluster.data import load_cadec, load_smm4h
from ghostcluster.scoring import consensus_gap, top_k_similarities
from ghostcluster.stats import paired_auprc_difference

NAMING_ENCODER = "biolord"   # used only to list the MedDRA terms nearest to a cluster centroid


def load(name):
    return np.load(os.path.join(OUT_DIR, name))


def text_consensus(corpus):
    """Consensus gap of every distinct expression of a corpus (label-free)."""
    gaps = []
    for e in ENCODERS:
        top, _ = top_k_similarities(load(f"emb_{e}_{corpus}.npy"), load(f"emb_{e}_meddra.npy"), k=5)
        gaps.append(1 - top.mean(1))
    return consensus_gap(gaps)


def nearest_terms(E, labels, clusters, terms, n=3):
    """MedDRA terms nearest to each cluster centroid; E rows are aligned with labels."""
    M = load(f"emb_{NAMING_ENCODER}_meddra.npy")
    out = {}
    for c in clusters:
        v = E[np.asarray(labels) == c].mean(0)
        _, arg = top_k_similarities((v / np.linalg.norm(v))[None, :], M, k=n)
        out[c] = "; ".join(f"{terms.name[j]} ({terms.level[j]})" for j in arg[0])
    return out


def describe(table, members, E, labels, terms, labelled=True):
    """One row per Ghost Cluster, most distant first."""
    ghosts = table[table.ghost].sort_values("mean", ascending=False)
    names = nearest_terms(E, labels, ghosts.index, terms)
    rows = []
    for c, r in ghosts.iterrows():
        m = members[members.cluster == c]
        row = dict(cluster=int(c), expressions=int(r["size"]), mentions=len(m), gap=float(r["mean"]),
                   examples=" | ".join(m.text.value_counts().index[:3]), nearest_meddra=names[c])
        if labelled:
            row["share_unmappable"] = float(m.nil.mean())
            coded = m.gold.dropna() if "gold" in m else pd.Series(dtype=float)
            row["concept_coherence"] = float(coded.value_counts(normalize=True).iloc[0]) if len(coded) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def smm4h(terms):
    smm = load_smm4h()
    texts = pd.read_csv(os.path.join(OUT_DIR, "texts_smm4h.csv")).text.astype(str).tolist()
    ti = smm.text.map({t: i for i, t in enumerate(texts)}).values
    E = load(f"emb_{CLUSTER_ENCODER}_smm4h.npy")[ti]
    gap = text_consensus("smm4h")[ti]
    X = umap_project(E, SEED)
    grid = []
    for mcs, ms, sel in itertools.product(*HDBSCAN_GRID.values()):
        lab = hdbscan_labels(X, min_cluster_size=mcs, min_samples=ms, cluster_selection_method=sel)
        keep = lab >= 0
        grid.append(dict(min_cluster_size=mcs, min_samples=ms, selection=sel, clusters=n_clusters(lab),
                         noise=1 - keep.mean(), silhouette=silhouette_score(X[keep], lab[keep]), dbcv=dbcv(X, lab)))
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(OUT_DIR, "smm4h_grid.csv"), index=False)
    best = grid.loc[grid.dbcv.idxmax()]
    chosen = grid[(grid.min_cluster_size == HDBSCAN_PARAMS["min_cluster_size"]) &
                  (grid.min_samples == HDBSCAN_PARAMS["min_samples"]) &
                  (grid.selection == HDBSCAN_PARAMS["cluster_selection_method"])].iloc[0]
    print(f"SMM4H ({len(smm)} mentions): DBCV-best = mcs {best.min_cluster_size}, ms {best.min_samples}, {best.selection} "
          f"(DBCV {best.dbcv:.3f}, {best.clusters} clusters); configured = DBCV {chosen.dbcv:.3f}, "
          f"{chosen.clusters} clusters, noise {chosen.noise:.1%}, silhouette {chosen.silhouette:.3f}")
    lab = hdbscan_labels(X)
    table = ghost_clusters(lab, gap)
    members = pd.DataFrame({"cluster": lab, "text": smm.text})
    ghosts = describe(table, members, load(f"emb_{NAMING_ENCODER}_smm4h.npy")[ti], lab, terms, labelled=False)
    ghosts.to_csv(os.path.join(OUT_DIR, "ghost_clusters_smm4h_hdbscan.csv"), index=False)
    print(f"  Ghost Clusters {int(table.ghost.sum())}/{len(table)}; most distant:")
    for _, r in ghosts.head(8).iterrows():
        print(f"    size {r.expressions:3d}  {r.examples}")


def labelled(terms, rng):
    scoring = []
    cadec_gold = load_cadec()[["text", "gold"]].assign(text=lambda d: d.text.astype(str))
    for corpus in ["cadec", "psytar"]:
        sc = pd.read_csv(os.path.join(OUT_DIR, f"scores_{corpus}.csv"))
        sc["text"] = sc.text.astype(str)
        if corpus == "cadec":
            sc["gold"] = cadec_gold.gold.values
        texts = pd.read_csv(os.path.join(OUT_DIR, f"texts_{corpus}.csv")).text.astype(str).tolist()
        E = load(f"emb_{CLUSTER_ENCODER}_{corpus}.npy")
        gap = sc.groupby("text")[PRIMARY].mean().reindex(texts).values
        k = ward_k(len(texts))
        partitions = {"ward": ward_labels(E, k), "hdbscan": hdbscan_after_umap(E, SEED)}
        y, g = sc.nil.values, sc.group.astype(str).values
        for method, lab in partitions.items():
            sc["cluster"] = sc.text.map(dict(zip(texts, lab)))
            print(f"\n{corpus.upper()} {method}: {n_clusters(lab)} clusters on {len(texts)} distinct expressions, "
                  f"noise {np.mean(lab == -1):.1%}")
            for key in [PRIMARY, "hybrid"]:
                s = sc[key].values
                mean = sc[sc.cluster >= 0].groupby("cluster")[key].mean()
                smooth = np.where(sc.cluster >= 0, sc.cluster.map(mean).fillna(0).values, s)
                blend = 0.5 * s + 0.5 * smooth
                m, lo, hi, _ = paired_auprc_difference(y, blend, s, g, rng)
                scoring.append(dict(corpus=corpus, clustering=method, score=key,
                                    auprc_expression=average_precision_score(y, s),
                                    auprc_blend=average_precision_score(y, blend), gain=m, lo=lo, hi=hi))
                print(f"  {key}: AUPRC expression {average_precision_score(y, s):.3f}, expression+cluster "
                      f"{average_precision_score(y, blend):.3f}; difference {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            table = ghost_clusters(lab, gap)
            ghosts = describe(table, sc, load(f"emb_{NAMING_ENCODER}_{corpus}.npy"), lab, terms)
            ghosts.to_csv(os.path.join(OUT_DIR, f"ghost_clusters_{corpus}_{method}.csv"), index=False)
            for _, r in ghosts.head(6).iterrows():
                print(f"    {r.expressions:3d} expr  unmappable {r.share_unmappable:4.0%}  {r.examples}  ->  {r.nearest_meddra}")
    pd.DataFrame(scoring).to_csv(os.path.join(OUT_DIR, "results_cluster_scoring.csv"), index=False)


def main():
    terms = pd.read_csv(os.path.join(OUT_DIR, "meddra_terms.csv"))
    smm4h(terms)
    labelled(terms, np.random.default_rng(SEED))


if __name__ == "__main__":
    main()
