"""Step 3 - clustering configuration, cluster-level scoring and Ghost Cluster content.

A. SMM4H-2017: silhouette and DBCV for the 24-configuration HDBSCAN grid; most distant Ghost Clusters.
B. CADEC and PsyTAR: cluster the labelled expressions; compare cluster-averaged with expression-level
   scores; list the most distant Ghost Clusters with the share of members judged unmappable.
C. Lexicon-based relevance filter applied to the Ghost Clusters (explored, not adopted).

Outputs (outputs/): smm4h_grid.csv, ghost_clusters_<corpus>.csv, results_cluster_scoring.csv, results_crf.csv
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ghostcluster import crf
from ghostcluster.clustering import dbcv, ghost_clusters, hdbscan_labels, n_clusters, umap_project
from ghostcluster.config import CLUSTER_ENCODER, HDBSCAN_GRID, HDBSCAN_PARAMS, OUT_DIR, SEED
from ghostcluster.data import load_smm4h
from ghostcluster.scoring import semantic_gap
from ghostcluster.stats import paired_auprc_difference

SEMANTIC = f"{CLUSTER_ENCODER}|top5|PT+LLT"


def load(name):
    return np.load(os.path.join(OUT_DIR, name))


def describe(table, members, label_share=None):
    """Attach size, representative expressions and (optionally) unmappable share to a ghost table."""
    rows = []
    for c, r in table[table.ghost].sort_values("mean", ascending=False).iterrows():
        m = members[members.cluster == c]
        row = dict(cluster=int(c), size=int(r["size"]), gap=float(r["mean"]),
                   examples=" | ".join(m.text.value_counts().index[:3]))
        if label_share is not None:
            row["share_unmappable"] = float(m.nil.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def smm4h():
    smm = load_smm4h()
    texts = pd.read_csv(os.path.join(OUT_DIR, "texts_smm4h.csv")).text.astype(str).tolist()
    ti = smm.text.map({t: i for i, t in enumerate(texts)}).values
    E = load(f"emb_{CLUSTER_ENCODER}_smm4h.npy")[ti]
    gap = semantic_gap(load(f"emb_{CLUSTER_ENCODER}_smm4h.npy"), load(f"emb_{CLUSTER_ENCODER}_meddra.npy"))[ti]
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
    ghosts = describe(table, pd.DataFrame({"cluster": lab, "text": smm.text}))
    ghosts.to_csv(os.path.join(OUT_DIR, "ghost_clusters_smm4h.csv"), index=False)
    print(f"  Ghost Clusters {int(table.ghost.sum())}/{len(table)}; most distant:")
    for _, r in ghosts.head(8).iterrows():
        print(f"    size {r['size']:3d}  {r.examples}")


def labelled(rng):
    scoring, crf_rows = [], []
    for corpus in ["cadec", "psytar"]:
        sc = pd.read_csv(os.path.join(OUT_DIR, f"scores_{corpus}.csv"))
        sc["text"] = sc.text.astype(str)
        texts = pd.read_csv(os.path.join(OUT_DIR, f"texts_{corpus}.csv")).text.astype(str).tolist()
        X = umap_project(load(f"emb_{CLUSTER_ENCODER}_{corpus}.npy"), SEED)
        lab = hdbscan_labels(X)
        sc["cluster"] = sc.text.map(dict(zip(texts, lab)))
        y, g = sc.nil.values, sc.group.astype(str).values
        print(f"\n{corpus.upper()}: {n_clusters(lab)} clusters on {len(texts)} distinct expressions, "
              f"noise {np.mean(lab == -1):.1%}, DBCV {dbcv(X, lab):.3f}")
        for key in [SEMANTIC, "hybrid"]:
            s = sc[key].values
            mean = sc[sc.cluster >= 0].groupby("cluster")[key].mean()
            smooth = np.where(sc.cluster >= 0, sc.cluster.map(mean).fillna(0).values, s)
            blend = 0.5 * s + 0.5 * smooth
            m, lo, hi = paired_auprc_difference(y, blend, s, g, rng)
            scoring.append(dict(corpus=corpus, score=key, auprc_expression=average_precision_score(y, s),
                                auprc_blend=average_precision_score(y, blend), gain=m, lo=lo, hi=hi))
            print(f"  {key}: AUPRC expression {average_precision_score(y, s):.3f}, expression+cluster "
                  f"{average_precision_score(y, blend):.3f}; difference {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        per_text = sc.groupby("text")[SEMANTIC].mean()
        table = ghost_clusters(lab, per_text.reindex(texts).values)
        ghosts = describe(table, sc, label_share=True)
        ghosts.to_csv(os.path.join(OUT_DIR, f"ghost_clusters_{corpus}.csv"), index=False)
        for _, r in ghosts.head(6).iterrows():
            print(f"    size {r['size']:3d}  unmappable {r.share_unmappable:4.0%}  {r.examples}")
        for c in table.index[table.ghost]:
            members = sc[sc.cluster == c]
            crf_rows.append(dict(corpus=corpus, cluster=int(c), kept=crf.cluster_passes(members.text),
                                 mentions=len(members), unmappable=int(members.nil.sum())))
        r = pd.DataFrame([x for x in crf_rows if x["corpus"] == corpus])
        kept, removed = r[r.kept], r[~r.kept]
        print(f"  relevance filter: keeps {len(kept)}/{len(r)} Ghost Clusters; unmappable share "
              f"kept {kept.unmappable.sum() / max(kept.mentions.sum(), 1):.1%} vs removed "
              f"{removed.unmappable.sum() / max(removed.mentions.sum(), 1):.1%}")
    pd.DataFrame(scoring).to_csv(os.path.join(OUT_DIR, "results_cluster_scoring.csv"), index=False)
    pd.DataFrame(crf_rows).to_csv(os.path.join(OUT_DIR, "results_crf.csv"), index=False)


def main():
    smm4h()
    labelled(np.random.default_rng(SEED))


if __name__ == "__main__":
    main()
