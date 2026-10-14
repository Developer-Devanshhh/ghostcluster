"""Step 6 - hidden-concept simulation on SMM4H-2017 and CADEC (Section 5.5).

Concepts are removed from MedDRA together with all of their LLTs, and every (expression, gold concept)
pair is scored against the reduced terminology; pairs whose gold PT was removed are the positives.
Three regimes are compared:
  PT     - single PTs are removed while their siblings stay (a near-neighbour concept usually remains);
  HLT    - whole High Level Terms are removed with every PT linked to them;
  HLGT   - whole High Level Group Terms are removed with every PT linked to them.
Ten random removals per regime and fraction; scores are the top-5 gap of each encoder and their consensus,
with string matching and dense normaliser confidence (SapBERT top-1) as baselines.

Output (outputs/): results_hidden_concepts.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ghostcluster.config import ENCODERS, OUT_DIR, PRIMARY
from ghostcluster.data import load_cadec, load_meddra, load_meddra_hierarchy, load_smm4h, norm
from ghostcluster.scoring import consensus_gap, top_k_similarities

POOL, K, REPS, FRACTIONS = 400, 5, 10, (0.1, 0.2, 0.3)


def load(name):
    return np.load(os.path.join(OUT_DIR, name))


def pairs_for(corpus, df, terms, llt2pt):
    level = dict(zip(terms.code, terms.level))
    parent = dict(zip(llt2pt.llt, llt2pt.pt))
    df = df.assign(gold_pt=[c if level.get(c) == "PT" else parent.get(c, np.nan) for c in df.gold])
    texts = pd.read_csv(os.path.join(OUT_DIR, f"texts_{corpus}.csv")).text.astype(str).tolist()
    df["ti"] = df.text.astype(str).map({t: i for i, t in enumerate(texts)})
    return df.dropna(subset=["gold_pt"]).drop_duplicates(["ti", "gold_pt"]).reset_index(drop=True)


def topk_after_removal(Q, M, top, arg, removed_rows):
    """Top-K similarities after deleting terminology rows, from a precomputed pool (exact fallback)."""
    keep = ~np.isin(arg, removed_rows)
    out = np.empty(len(Q))
    short = keep.sum(1) < K
    for i in np.where(~short)[0]:
        out[i] = top[i][keep[i]][:K].mean()
    if short.any():
        mask = np.ones(len(M), bool)
        mask[removed_rows] = False
        t, _ = top_k_similarities(Q[short], M[mask], k=K)
        out[short] = t.mean(1)
    return 1 - out


STRING_POOL = 50
STRING, DENSE = "string", "dense normaliser (SapBERT top-1)"


def string_pool(texts, names):
    """The STRING_POOL best token-sort matches of each text among all term names (similarity, row index)."""
    choices = [norm(n) for n in names]
    sims = np.zeros((len(texts), STRING_POOL), dtype=np.float32)
    rows = np.zeros((len(texts), STRING_POOL), dtype=np.int64)
    for i in range(0, len(texts), 200):
        S = process.cdist([norm(x) for x in texts[i:i + 200]], choices, scorer=fuzz.token_sort_ratio,
                          workers=-1, dtype=np.uint8)
        a = np.argpartition(-S, STRING_POOL, axis=1)[:, :STRING_POOL]
        v = np.take_along_axis(S, a, 1)
        o = np.argsort(-v, axis=1)
        sims[i:i + 200], rows[i:i + 200] = np.take_along_axis(v, o, 1), np.take_along_axis(a, o, 1)
    return sims, rows


def string_after_removal(texts, names, sims, rows, removed_rows):
    keep = ~np.isin(rows, removed_rows)
    best = np.where(keep.any(1), np.where(keep, sims, -1).max(1), np.nan)
    missing = np.where(np.isnan(best))[0]
    if len(missing):
        mask = np.ones(len(names), bool)
        mask[removed_rows] = False
        allowed = [norm(n) for n in np.asarray(names)[mask]]
        for i in missing:
            best[i] = process.extractOne(norm(texts[i]), allowed, scorer=fuzz.token_sort_ratio)[1]
    return 1 - best / 100.0


def dense_after_removal(top, arg, removed_rows):
    keep = ~np.isin(arg, removed_rows)
    first = np.where(keep.any(1), np.argmax(keep, 1), -1)
    return 1 - np.where(first >= 0, top[np.arange(len(top)), np.maximum(first, 0)], 0.0)


def main():
    _, llt2pt = load_meddra()
    terms = pd.read_csv(os.path.join(OUT_DIR, "meddra_terms.csv"))
    hier = load_meddra_hierarchy()
    parent = dict(zip(llt2pt.llt, llt2pt.pt))
    row_pt = np.array([c if lvl == "PT" else parent.get(c, -1) for c, lvl in zip(terms.code, terms.level)])
    rows = []
    for corpus, df in [("smm4h", load_smm4h()), ("cadec", load_cadec().query("nil == 0"))]:
        pairs = pairs_for(corpus, df, terms, llt2pt)
        golds = pairs.gold_pt.unique()
        print(f"{corpus}: {len(pairs)} distinct (expression, gold PT) pairs, {len(golds)} gold PTs")
        pools = {}
        for e in ENCODERS:
            Q, M = load(f"emb_{e}_{corpus}.npy")[pairs.ti.values], load(f"emb_{e}_meddra.npy")
            top, arg = top_k_similarities(Q, M, k=POOL)
            pools[e] = (Q, M, top, arg)
        texts = pd.read_csv(os.path.join(OUT_DIR, f"texts_{corpus}.csv")).text.astype(str).values[pairs.ti.values]
        s_sims, s_rows = string_pool(texts, terms.name.values)
        for regime in ["PT", "HLT", "HLGT"]:
            if regime == "PT":
                units = golds
            else:
                col = regime.lower()
                units = hier[hier.pt.isin(golds)][col].unique()
            for frac in FRACTIONS:
                res = {e: [] for e in list(ENCODERS) + [PRIMARY, STRING, DENSE]}
                prev = []
                for rep in range(REPS):
                    rng = np.random.default_rng(rep)
                    chosen = rng.choice(units, max(1, int(round(frac * len(units)))), replace=False)
                    hidden_pts = set(chosen) if regime == "PT" else set(hier[hier[regime.lower()].isin(chosen)].pt)
                    y = pairs.gold_pt.isin(hidden_pts).values
                    removed = np.where(np.isin(row_pt, list(hidden_pts)))[0]
                    gaps = {e: topk_after_removal(*pools[e], removed) for e in ENCODERS}
                    gaps[PRIMARY] = consensus_gap(list(gaps.values()))
                    gaps[STRING] = string_after_removal(texts, terms.name.values, s_sims, s_rows, removed)
                    gaps[DENSE] = dense_after_removal(pools["sapbert"][2], pools["sapbert"][3], removed)
                    for e, g in gaps.items():
                        res[e].append((roc_auc_score(y, g), average_precision_score(y, g), y.mean()))
                    prev.append(y.mean())
                for e, v in res.items():
                    v = np.array(v)
                    rows.append(dict(corpus=corpus, regime=regime, fraction=frac, encoder=e,
                                     positives=v[:, 2].mean(), auroc=v[:, 0].mean(), auroc_sd=v[:, 0].std(),
                                     auprc=v[:, 1].mean(), auprc_sd=v[:, 1].std()))
                c = next(r for r in rows[::-1] if r["encoder"] == PRIMARY)
                s = next(r for r in rows[::-1] if r["encoder"] == STRING)
                print(f"  {regime:4s} {int(frac * 100)}%: positives {np.mean(prev):.1%}  consensus AUROC "
                      f"{c['auroc']:.3f} +/- {c['auroc_sd']:.3f}  AUPRC {c['auprc']:.3f}  (string AUROC {s['auroc']:.3f})")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "results_hidden_concepts.csv"), index=False)


if __name__ == "__main__":
    main()
