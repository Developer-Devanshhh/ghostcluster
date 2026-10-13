"""Step 2 - score expressions and evaluate them against independent labels.

A. Descriptive statistics and verbatim terminology coverage (Section 5.1).
B. Human no-concept labels: CADEC (MedDRA CONCEPT_LESS) and PsyTAR (no SNOMED CT code) (Sections 5.2-5.3).
C. SMM4H-2017: mapping-difficulty proxy (Section 6.4) and hidden-concept simulation (Section 5.4).

Outputs (outputs/): descriptives.json, coverage.csv, scores_<corpus>.csv, results_human_labels.csv,
results_gain_over_string.csv, results_hidden_concepts.csv, results_mapping_difficulty.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ghostcluster.config import ENCODERS, HYBRID_ENCODERS, OUT_DIR, SEED
from ghostcluster.data import gold_pt, load_cadec, load_meddra, load_psytar, load_smm4h, norm
from ghostcluster.scoring import lexical_gap, top_k_similarities, word_count
from ghostcluster.stats import bootstrap_metrics, paired_auprc_difference

LENGTH, STRING, HYBRID = "length", "string", "hybrid"
STOPWORDS = set("a an the of in on at to and or is are was were be been my me i im it its with for from by that this as up so".split())


def method_keys():
    keys = [LENGTH, STRING]
    for e in ENCODERS:
        keys += [f"{e}|max|PT", f"{e}|max|PT+LLT", f"{e}|top5|PT+LLT"]
    return keys + [HYBRID]


def load(name):
    return np.load(os.path.join(OUT_DIR, name))


def descriptives(terms):
    """Corpus sizes and verbatim coverage of distinct expressions by MedDRA term names."""
    stem = PorterStemmer().stem
    key = lambda s: " ".join(sorted({stem(t) for t in s.split() if t not in STOPWORDS}))
    names = terms.assign(n=terms.name.map(norm))
    pt, llt = set(names.n[names.level == "PT"]), set(names.n[names.level == "LLT"])
    stemmed = set(names.n.map(key))
    cadec, psytar, smm4h = load_cadec(), load_psytar(), load_smm4h()
    rows = []
    for corpus, texts in [("SMM4H-2017", smm4h.text), ("CADEC", cadec.text), ("PsyTAR", psytar.text)]:
        u = pd.Series(texts.map(norm).unique())
        rows.append(dict(corpus=corpus, distinct=len(u), pt=u.isin(pt).mean(), llt=u.isin(llt).mean(),
                         pt_or_llt=(u.isin(pt) | u.isin(llt)).mean(), stemmed=u.map(key).isin(stemmed).mean()))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "coverage.csv"), index=False)

    any_level = names.groupby("n")["level"].agg(set).to_dict()
    unc = psytar[psytar.nil == 1].text.map(norm)
    verbatim = unc[unc.isin(any_level)]
    smm_level = smm4h.gold.map(dict(zip(terms.code, terms.level)))
    out = dict(
        cadec_mentions=len(cadec), cadec_posts=int(cadec.post_id.nunique()), cadec_nil=int(cadec.nil.sum()),
        psytar_mentions=len(psytar), psytar_reviews=int(psytar.review_id.nunique()), psytar_nil=int(psytar.nil.sum()),
        smm4h_mentions=len(smm4h), smm4h_distinct_strings=int(smm4h.text.nunique()),
        smm4h_gold_levels=smm_level.fillna("none").value_counts().to_dict(),
        smm4h_share_gold_pt=float((smm_level == "PT").mean()),
        psytar_uncoded_verbatim_meddra=int(len(verbatim)),
        psytar_uncoded_verbatim_llt_only=int(sum(any_level[v] == {"LLT"} for v in verbatim)),
        psytar_uncoded_verbatim_examples=verbatim.value_counts().head(10).to_dict(),
    )
    return out


def human_label_evaluation(terms, rng):
    is_pt = (terms.level == "PT").values
    results, gains = [], []
    for corpus, df, group in [("cadec", load_cadec(), "post_id"), ("psytar", load_psytar(), "review_id")]:
        texts = pd.read_csv(os.path.join(OUT_DIR, f"texts_{corpus}.csv")).text.astype(str).tolist()
        ix = df.text.astype(str).map({t: i for i, t in enumerate(texts)}).values
        y, g = df.nil.values, df[group].astype(str).values
        f = {LENGTH: word_count(df.text), STRING: lexical_gap(texts, terms.name)[ix]}
        for e in ENCODERS:
            Q, M = load(f"emb_{e}_{corpus}.npy"), load(f"emb_{e}_meddra.npy")
            top_pt, _ = top_k_similarities(Q, M[is_pt], k=1)
            top_all, _ = top_k_similarities(Q, M, k=5)
            f[f"{e}|max|PT"] = 1 - top_pt[ix, 0]
            f[f"{e}|max|PT+LLT"] = 1 - top_all[ix, 0]
            f[f"{e}|top5|PT+LLT"] = 1 - top_all[ix].mean(1)
        # Hybrid (Eq. 2): logistic model, five-fold cross-validation grouped by document.
        X = np.column_stack([f[LENGTH], f[STRING]] + [f[f"{e}|max|PT+LLT"] for e in HYBRID_ENCODERS])
        f[HYBRID] = np.zeros(len(y))
        for train, test in GroupKFold(n_splits=5).split(X, y, g):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            f[HYBRID][test] = model.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        scores = pd.DataFrame(f)
        scores.insert(0, "text", df.text.values); scores.insert(1, "nil", y); scores.insert(2, "group", g)
        scores.to_csv(os.path.join(OUT_DIR, f"scores_{corpus}.csv"), index=False)

        print(f"\n{corpus.upper()}: {len(y)} mentions, {y.sum()} without concept ({y.mean():.1%})")
        for k in method_keys():
            r = bootstrap_metrics(y, f[k], g, rng)
            results.append(dict(corpus=corpus, method=k, prevalence=y.mean(), **r))
            print(f"  {k:24s} AUROC {r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}]  "
                  f"AUPRC {r['auprc']:.3f} [{r['auprc_lo']:.3f}, {r['auprc_hi']:.3f}]")
        for k in method_keys()[2:]:
            m, lo, hi = paired_auprc_difference(y, f[k], f[STRING], g, rng)
            gains.append(dict(corpus=corpus, method=k, gain=m, lo=lo, hi=hi))
            print(f"  AUPRC gain over string, {k:24s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "results_human_labels.csv"), index=False)
    pd.DataFrame(gains).to_csv(os.path.join(OUT_DIR, "results_gain_over_string.csv"), index=False)


def smm4h_analyses(terms, llt2pt):
    smm = load_smm4h()
    smm["gold_pt"] = gold_pt(smm, terms, llt2pt)
    texts = pd.read_csv(os.path.join(OUT_DIR, "texts_smm4h.csv")).text.astype(str).tolist()
    smm["ti"] = smm.text.map({t: i for i, t in enumerate(texts)})
    is_pt = (terms.level == "PT").values
    pt_rows = np.where(is_pt)[0]
    pt_codes = terms.code.values[pt_rows]
    row_of = dict(zip(pt_codes, pt_rows))
    parent = dict(zip(llt2pt.llt, llt2pt.pt))
    llt_parent = terms.code.map(parent).values
    v = smm[smm.gold_pt.isin(row_of)].copy()
    print(f"\nSMM4H: {len(smm)} mentions; {len(v)} with a PT-level gold concept (after LLT->PT mapping)")
    proxy, hidden = [], []
    for e in ENCODERS:
        Q, M = load(f"emb_{e}_smm4h.npy"), load(f"emb_{e}_meddra.npy")
        # Mapping-difficulty proxy (Section 6.4): coupled to the detector by construction.
        top, _ = top_k_similarities(Q, M[pt_rows], k=1)
        sigma = top[v.ti.values, 0]
        assigned = np.einsum("ij,ij->i", Q[v.ti.values], M[[row_of[c] for c in v.gold_pt]])
        rho = spearmanr(sigma, 1 - assigned)[0]
        proxy.append(dict(encoder=e, spearman=rho, auroc=roc_auc_score(assigned < np.median(assigned), -sigma)))
        print(f"  [{e}] mapping-difficulty proxy: Spearman {rho:.3f}")
        # Hidden-concept simulation (Section 5.4).
        pairs = v.drop_duplicates(["ti", "gold_pt"])
        golds = pairs.gold_pt.unique()
        for frac in (0.1, 0.2, 0.3):
            aucs = []
            for rep in range(10):
                hid = set(np.random.default_rng(rep).choice(golds, int(frac * len(golds)), replace=False))
                keep = ~np.isin(terms.code.values, list(hid)) & ~np.isin(llt_parent, list(hid))
                top, _ = top_k_similarities(Q[pairs.ti.values], M[keep], k=1)
                aucs.append(roc_auc_score(pairs.gold_pt.isin(hid).values, -top[:, 0]))
            hidden.append(dict(encoder=e, hidden=frac, auroc=np.mean(aucs), auroc_sd=np.std(aucs)))
            print(f"  [{e}] hidden {int(frac * 100)}%: AUROC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    pd.DataFrame(proxy).to_csv(os.path.join(OUT_DIR, "results_mapping_difficulty.csv"), index=False)
    pd.DataFrame(hidden).to_csv(os.path.join(OUT_DIR, "results_hidden_concepts.csv"), index=False)


def main():
    terms, llt2pt = load_meddra()
    terms = pd.read_csv(os.path.join(OUT_DIR, "meddra_terms.csv"))     # embedding row order
    d = descriptives(terms)
    with open(os.path.join(OUT_DIR, "descriptives.json"), "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, default=str)
    print(json.dumps(d, indent=2, default=str))
    human_label_evaluation(terms, np.random.default_rng(SEED))
    smm4h_analyses(terms, llt2pt)


if __name__ == "__main__":
    main()
