"""Step 2 - score expressions and evaluate them against independent labels.

A. Descriptive statistics and verbatim terminology coverage (Section 5.1).
B. Human no-concept labels: CADEC (MedDRA CONCEPT_LESS) and PsyTAR (no SNOMED CT code) (Sections 5.2-5.3).
   The primary score is the label-free consensus gap; the supervised hybrid is a secondary analysis.
C. SMM4H-2017: mapping-difficulty proxy (Section 6.4).

Outputs (outputs/): descriptives.json, coverage.csv, scores_<corpus>.csv, results_human_labels.csv,
results_gain_over_string.csv, results_primary_comparisons.csv, results_mapping_difficulty.csv
"""
import glob
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
from ghostcluster.config import CADEC_DIR, ENCODERS, HYBRID_ENCODERS, OUT_DIR, PRIMARY, PSYTAR_XLSX, SEED
from ghostcluster.data import gold_pt, load_cadec, load_meddra, load_psytar, load_smm4h, norm
from ghostcluster.scoring import consensus_gap, lexical_gap, top_k_similarities, word_count
from ghostcluster.stats import bootstrap_metrics, paired_auprc_difference

LENGTH, STRING, HYBRID = "length", "string", "hybrid"
STOPWORDS = set("a an the of in on at to and or is are was were be been my me i im it its with for from by that this as up so".split())


def method_keys():
    keys = [LENGTH, STRING]
    for e in ENCODERS:
        keys += [f"{e}|max|PT", f"{e}|max|PT+LLT", f"{e}|top5|PT+LLT"]
    return keys + [PRIMARY, HYBRID]


def load(name):
    return np.load(os.path.join(OUT_DIR, name))


def hybrid_features(f, encoders=HYBRID_ENCODERS, length=True):
    cols = ([f[LENGTH]] if length else []) + [f[STRING]] + [f[f"{e}|max|PT+LLT"] for e in encoders]
    return np.column_stack(cols)


def hybrid_model():
    """Class-balanced logistic regression; the scaler is fitted inside each training fold."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def cross_validated(X, y, groups):
    """Out-of-fold probabilities, five folds grouped so that no group is split between train and test."""
    out = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        out[test] = hybrid_model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return out


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

    def verbatim(df):
        unc = df[df.nil == 1].text.map(norm)
        v = unc[unc.isin(any_level)]
        return int(len(v)), int(sum(any_level[t] == {"LLT"} for t in v)), v.value_counts().head(10).to_dict()

    p_n, p_llt, p_ex = verbatim(psytar)
    c_n, c_llt, c_ex = verbatim(cadec)
    # What "no SNOMED CT code" means in PsyTAR: the UMLS concept chosen by the annotators has no SNOMED CT
    # code (UMLS1 is a CUI) and, usually, a related SNOMED CT concept is given as a fallback (SNOMED-CT.1).
    raw = pd.read_excel(PSYTAR_XLSX, sheet_name="ADR_Mapped")
    raw = raw[~raw["SNOMED-CT"].astype(str).str.lower().str.contains("snomedct", na=False)]
    has_cui = raw.UMLS1.astype(str).str.strip().str.match(r"C\d{7}")
    has_fallback = raw["SNOMED-CT.1"].astype(str).str.contains("SNOMEDCT", case=False)
    smm_level = smm4h.gold.map(dict(zip(terms.code, terms.level)))
    cadec_files = glob.glob(os.path.join(CADEC_DIR, "**", "meddra", "*.ann"), recursive=True)
    return dict(
        cadec_mentions=len(cadec), cadec_posts=int(cadec.post_id.nunique()), cadec_posts_total=len(cadec_files),
        cadec_nil=int(cadec.nil.sum()),
        psytar_mentions=len(psytar), psytar_reviews=int(psytar.review_id.nunique()), psytar_nil=int(psytar.nil.sum()),
        smm4h_mentions=len(smm4h), smm4h_distinct_strings=int(smm4h.text.nunique()),
        smm4h_distinct_normalised=int(smm4h.text.map(norm).nunique()),
        smm4h_gold_levels=smm_level.fillna("none").value_counts().to_dict(),
        smm4h_share_gold_pt=float((smm_level == "PT").mean()),
        psytar_uncoded_with_umls_concept=int(has_cui.sum()),
        psytar_uncoded_with_fallback_snomed=int(has_fallback.sum()),
        psytar_uncoded_without_any_concept=int((~has_cui & ~has_fallback).sum()),
        psytar_uncoded_verbatim_meddra=p_n, psytar_uncoded_verbatim_llt_only=p_llt, psytar_uncoded_verbatim_examples=p_ex,
        cadec_conceptless_verbatim_meddra=c_n, cadec_conceptless_verbatim_llt_only=c_llt,
        cadec_conceptless_verbatim_examples=c_ex,
    )


def corpus_features(corpus, df, terms):
    """All expression-level scores for one labelled corpus (one row per mention)."""
    is_pt = (terms.level == "PT").values
    texts = pd.read_csv(os.path.join(OUT_DIR, f"texts_{corpus}.csv")).text.astype(str).tolist()
    ix = df.text.astype(str).map({t: i for i, t in enumerate(texts)}).values
    f = {LENGTH: word_count(df.text), STRING: lexical_gap(texts, terms.name)[ix]}
    top5 = []
    for e in ENCODERS:
        Q, M = load(f"emb_{e}_{corpus}.npy"), load(f"emb_{e}_meddra.npy")
        top_pt, _ = top_k_similarities(Q, M[is_pt], k=1)
        top_all, _ = top_k_similarities(Q, M, k=5)
        f[f"{e}|max|PT"] = 1 - top_pt[ix, 0]
        f[f"{e}|max|PT+LLT"] = 1 - top_all[ix, 0]
        f[f"{e}|top5|PT+LLT"] = 1 - top_all[ix].mean(1)
        top5.append(1 - top_all.mean(1))
    # Primary score: consensus over the distinct expressions of the corpus, no labels involved.
    f[PRIMARY] = consensus_gap(top5)[ix]
    return f


def human_label_evaluation(terms, rng):
    results, gains, primary = [], [], []
    for corpus, df, group in [("cadec", load_cadec(), "post_id"), ("psytar", load_psytar(), "review_id")]:
        y, g = df.nil.values, df[group].astype(str).values
        f = corpus_features(corpus, df, terms)
        f[HYBRID] = cross_validated(hybrid_features(f), y, g)
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
            m, lo, hi, p = paired_auprc_difference(y, f[k], f[STRING], g, rng)
            gains.append(dict(corpus=corpus, method=k, gain=m, lo=lo, hi=hi, p=p))
            print(f"  AUPRC gain over string, {k:24s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        best = max((k for k in f if "|top5|" in k), key=lambda k: roc_auc_score(y, f[k]))
        for other in [HYBRID, best, LENGTH]:
            m, lo, hi, p = paired_auprc_difference(y, f[PRIMARY], f[other], g, rng)
            primary.append(dict(corpus=corpus, versus=other, gain=m, lo=lo, hi=hi, p=p))
            print(f"  consensus minus {other:24s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    gains = pd.DataFrame(gains)
    # Holm correction across all comparisons with the string baseline (both corpora).
    order = np.argsort(gains.p.values)
    adj = np.maximum.accumulate([min(1, p * (len(order) - i)) for i, p in enumerate(gains.p.values[order])])
    gains.loc[gains.index[order], "p_holm"] = adj
    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "results_human_labels.csv"), index=False)
    gains.to_csv(os.path.join(OUT_DIR, "results_gain_over_string.csv"), index=False)
    pd.DataFrame(primary).to_csv(os.path.join(OUT_DIR, "results_primary_comparisons.csv"), index=False)


def mapping_difficulty(terms, llt2pt):
    """Spearman correlation between detector similarity and similarity to the gold concept (circular proxy)."""
    smm = load_smm4h()
    smm["gold_pt"] = gold_pt(smm, terms, llt2pt)
    texts = pd.read_csv(os.path.join(OUT_DIR, "texts_smm4h.csv")).text.astype(str).tolist()
    smm["ti"] = smm.text.map({t: i for i, t in enumerate(texts)})
    pt_rows = np.where((terms.level == "PT").values)[0]
    row_of = dict(zip(terms.code.values[pt_rows], pt_rows))
    v = smm[smm.gold_pt.isin(row_of)].copy()
    print(f"\nSMM4H: {len(smm)} mentions; {len(v)} with a PT-level gold concept (after LLT->PT mapping)")
    proxy = []
    for e in ENCODERS:
        Q, M = load(f"emb_{e}_smm4h.npy"), load(f"emb_{e}_meddra.npy")
        top, _ = top_k_similarities(Q, M[pt_rows], k=1)
        sigma = top[v.ti.values, 0]
        assigned = np.einsum("ij,ij->i", Q[v.ti.values], M[[row_of[c] for c in v.gold_pt]])
        rho = spearmanr(sigma, 1 - assigned)[0]
        proxy.append(dict(encoder=e, spearman=rho, auroc=roc_auc_score(assigned < np.median(assigned), -sigma)))
        print(f"  [{e}] mapping-difficulty proxy: Spearman {rho:.3f}")
    pd.DataFrame(proxy).to_csv(os.path.join(OUT_DIR, "results_mapping_difficulty.csv"), index=False)


def main():
    terms, llt2pt = load_meddra()
    terms = pd.read_csv(os.path.join(OUT_DIR, "meddra_terms.csv"))     # embedding row order
    d = descriptives(terms)
    with open(os.path.join(OUT_DIR, "descriptives.json"), "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, default=str)
    print(json.dumps(d, indent=2, default=str))
    human_label_evaluation(terms, np.random.default_rng(SEED))
    mapping_difficulty(terms, llt2pt)


if __name__ == "__main__":
    main()
