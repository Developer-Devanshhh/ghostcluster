"""Step 5 - sensitivity analyses and additional baselines (Sections 5.3-5.4 and Supplementary Information).

A. Length: performance within length strata and after removing the linear effect of length (label-free).
B. Score definition: top-k for k in {1, 3, 5, 10}, concept-level aggregation (one similarity per parent PT),
   and performance by the number of LLTs under the nearest PT.
C. Baselines: TF-IDF (character and word n-grams), BM25, dense-normaliser confidence, and supervised
   classifiers (TF-IDF and frozen sentence embeddings) under the same grouped cross-validation.
D. Supervised hybrid: ablation of the length feature, folds grouped by expression, cross-corpus transfer;
   supervised classifiers versus the label-free consensus within and across corpora.
E. Operating points: precision at k, precision at fixed recall, share of mentions read to reach a recall.
F. Labels: distinct-expression evaluation; removal of positives that are verbatim MedDRA term names.

Outputs (outputs/): results_length.csv, results_score_variants.csv, results_baselines.csv,
results_hybrid_sensitivity.csv, results_supervised.csv, results_operating_points.csv, results_label_checks.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ghostcluster.config import ENCODERS, HYBRID_ENCODERS, OUT_DIR, PRIMARY, SEED
from ghostcluster.data import load_meddra, norm
from ghostcluster.scoring import concept_level_gap, consensus_gap, top_k_similarities
from ghostcluster.stats import bootstrap_metrics

CORPORA = ["cadec", "psytar"]
TOP5 = [f"{e}|top5|PT+LLT" for e in ENCODERS]


def load(name):
    return np.load(os.path.join(OUT_DIR, name))


def scores(corpus):
    sc = pd.read_csv(os.path.join(OUT_DIR, f"scores_{corpus}.csv"))
    sc["text"] = sc.text.astype(str)
    texts = pd.read_csv(os.path.join(OUT_DIR, f"texts_{corpus}.csv")).text.astype(str).tolist()
    sc["ti"] = sc.text.map({t: i for i, t in enumerate(texts)})
    return sc, texts


def metrics(y, s):
    return dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s))


def grouped_oof(X, y, groups, make):
    out = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        out[test] = make().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return out


def logistic():
    return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=3000, class_weight="balanced"))


# ---------------------------------------------------------------------------------------------
def length_analysis():
    rows = []
    for corpus in CORPORA:
        sc, _ = scores(corpus)
        y = sc.nil.values
        bins = pd.cut(sc.length, [0, 2, 4, np.inf], labels=["1-2 words", "3-4 words", "5+ words"])
        for b in bins.cat.categories:
            m = (bins == b).values
            for key in ["string", PRIMARY, "hybrid"]:
                rows.append(dict(corpus=corpus, stratum=b, method=key, n=int(m.sum()), prevalence=y[m].mean(),
                                 **metrics(y[m], sc[key].values[m])))
        # Label-free residualisation: remove the least-squares fit of the score on log length.
        L = np.column_stack([np.ones(len(sc)), np.log(sc.length.values)])
        for key in ["string", PRIMARY]:
            s = sc[key].values
            resid = s - L @ np.linalg.lstsq(L, s, rcond=None)[0]
            rows.append(dict(corpus=corpus, stratum="all, length removed", method=key, n=len(y),
                             prevalence=y.mean(), **metrics(y, resid)))
        # Incremental value over length in a supervised model (grouped five-fold CV).
        g = sc.group.astype(str).values
        for name, cols in [("length only", ["length"]), ("length + consensus", ["length", PRIMARY]),
                           ("length + string", ["length", "string"])]:
            p = grouped_oof(sc[cols].values, y, g, logistic)
            rows.append(dict(corpus=corpus, stratum="all, supervised", method=name, n=len(y),
                             prevalence=y.mean(), **metrics(y, p)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_length.csv"), index=False)
    print(out.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------
def score_variants(terms, llt2pt):
    parent_of = dict(zip(llt2pt.llt, llt2pt.pt))
    parent = np.array([c if lvl == "PT" else parent_of.get(c, c) for c, lvl in zip(terms.code, terms.level)])
    n_llt = pd.Series(parent[(terms.level == "LLT").values]).value_counts()
    rows = []
    for corpus in CORPORA:
        sc, texts = scores(corpus)
        y, ix = sc.nil.values, sc.ti.values
        variants = {}
        density = None
        for e in ENCODERS:
            Q, M = load(f"emb_{e}_{corpus}.npy"), load(f"emb_{e}_meddra.npy")
            top, arg = top_k_similarities(Q, M, k=10)
            for k in (1, 3, 5, 10):
                variants.setdefault(f"top-{k}", {})[e] = 1 - top[:, :k].mean(1)
            variants.setdefault("concept-level top-5", {})[e] = concept_level_gap(Q, M, parent, k=5)
            if e == "biolord":
                density = pd.Series(parent[arg[:, 0]]).map(n_llt).fillna(0).values
        for v, per_enc in variants.items():
            for e, gap in list(per_enc.items()) + [(PRIMARY, consensus_gap(list(per_enc.values())))]:
                rows.append(dict(corpus=corpus, variant=v, encoder=e, **metrics(y, gap[ix])))
        # Performance by terminology density: LLT count of the nearest concept (BioLORD), in tertiles.
        tert = pd.qcut(density[ix], 3, labels=["low", "middle", "high"], duplicates="drop")
        for t in tert.categories:
            m = (tert == t)
            rows.append(dict(corpus=corpus, variant=f"LLT density {t}", encoder=PRIMARY,
                             **metrics(y[m], sc[PRIMARY].values[m])))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_score_variants.csv"), index=False)
    print(out.pivot_table(index=["variant", "encoder"], columns="corpus", values=["auroc", "auprc"]).round(3).to_string())


# ---------------------------------------------------------------------------------------------
def bm25_best(texts, names, k1=1.2, b=0.75):
    """Highest BM25 score of each expression against the term names (terms as documents)."""
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    D = cv.fit_transform(names).tocsc().astype(np.float64)
    N, dl = D.shape[0], np.asarray(D.sum(1)).ravel()
    idf = np.log(1 + (N - np.bincount(D.indices, minlength=D.shape[1]) + 0.5) /
                 (np.bincount(D.indices, minlength=D.shape[1]) + 0.5))
    D = D.tocoo()
    w = D.data * (k1 + 1) / (D.data + k1 * (1 - b + b * dl[D.row] / dl.mean())) * idf[D.col]
    W = csr_matrix((w, (D.row, D.col)), shape=D.shape)
    Qb = (cv.transform(texts) > 0).astype(np.float64)
    best = np.zeros(len(texts))
    for i in range(0, len(texts), 500):
        best[i:i + 500] = (Qb[i:i + 500] @ W.T).max(1).toarray().ravel()
    return best


def tfidf_best(texts, names, **kw):
    v = TfidfVectorizer(**kw).fit(list(names) + list(texts))
    T, Q = v.transform(names), v.transform(texts)
    best = np.zeros(len(texts))
    for i in range(0, len(texts), 500):
        best[i:i + 500] = (Q[i:i + 500] @ T.T).max(1).toarray().ravel()
    return best


def baselines(terms, rng):
    names = [norm(n) for n in terms.name]
    rows, keep = [], {}
    for corpus in CORPORA:
        sc, texts = scores(corpus)
        y, g, ix = sc.nil.values, sc.group.astype(str).values, sc.ti.values
        nt = [norm(t) for t in texts]
        s = {
            "TF-IDF, character 3-5-grams": 1 - tfidf_best(nt, names, analyzer="char_wb", ngram_range=(3, 5))[ix],
            "TF-IDF, word 1-2-grams": 1 - tfidf_best(nt, names, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")[ix],
            "BM25": -bm25_best(nt, names)[ix],
            "Dense normaliser confidence (SapBERT, top-1)": sc["sapbert|max|PT+LLT"].values,
            "String match (token-sort ratio)": sc.string.values,
            "Consensus gap (label-free)": sc[PRIMARY].values,
            "Hybrid (supervised)": sc.hybrid.values,
        }
        text_x = np.array(sc.text.map(norm))
        s["Supervised TF-IDF classifier"] = grouped_oof(
            np.arange(len(y)).reshape(-1, 1), y, g,
            lambda: _TextModel(text_x))
        for e in ["sapbert", "biolord"]:
            X = load(f"emb_{e}_{corpus}.npy")[ix]
            s[f"Supervised embedding classifier ({e})"] = grouped_oof(X, y, g, logistic)
        for name, v in s.items():
            rows.append(dict(corpus=corpus, method=name, **bootstrap_metrics(y, v, g, rng)))
            print(f"{corpus:6s} {name:46s} AUROC {rows[-1]['auroc']:.3f} AUPRC {rows[-1]['auprc']:.3f}")
        keep[corpus] = s
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "results_baselines.csv"), index=False)
    return keep


class _TextModel:
    """Character n-gram TF-IDF + class-balanced logistic regression, fitted on row indices of `text`."""

    def __init__(self, text):
        self.text = text

    def fit(self, idx, y):
        self.v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        X = self.v.fit_transform(self.text[idx.ravel()])
        self.m = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(X, y)
        return self

    def predict_proba(self, idx):
        return self.m.predict_proba(self.v.transform(self.text[idx.ravel()]))


# ---------------------------------------------------------------------------------------------
def hybrid_sensitivity():
    data = {c: scores(c)[0] for c in CORPORA}
    full = ["length", "string"] + [f"{e}|max|PT+LLT" for e in HYBRID_ENCODERS]
    variants = {"hybrid (as specified)": full, "hybrid without length": full[1:],
                "hybrid on consensus": ["length", "string", PRIMARY]}
    rows = []
    for corpus, sc in data.items():
        y = sc.nil.values
        for name, cols in variants.items():
            for grouping, groups in [("documents", sc.group.astype(str).values),
                                     ("expressions", sc.text.map(norm).values)]:
                p = grouped_oof(sc[cols].values, y, groups, logistic)
                rows.append(dict(train=corpus, test=corpus, variant=name, folds=f"grouped by {grouping}", **metrics(y, p)))
    for src, tgt in [("cadec", "psytar"), ("psytar", "cadec")]:
        for name, cols in variants.items():
            m = logistic().fit(data[src][cols].values, data[src].nil.values)
            p = m.predict_proba(data[tgt][cols].values)[:, 1]
            rows.append(dict(train=src, test=tgt, variant=name, folds="none (transfer)", **metrics(data[tgt].nil.values, p)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_hybrid_sensitivity.csv"), index=False)
    print(out.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------
def supervised_versus_label_free():
    """Within-corpus (folds grouped by document or by expression) and cross-corpus performance."""
    data = {c: scores(c)[0] for c in CORPORA}
    text = {c: np.array(d.text.map(norm)) for c, d in data.items()}
    emb = {c: load(f"emb_biolord_{c}.npy")[d.ti.values] for c, d in data.items()}
    hyb = ["length", "string"] + [f"{e}|max|PT+LLT" for e in HYBRID_ENCODERS]
    methods = {
        "Supervised TF-IDF classifier": ("text", None),
        "Supervised embedding classifier (BioLORD-2023)": ("emb", None),
        "Hybrid (supervised)": ("cols", hyb),
        "Consensus gap (label-free)": ("none", None),
    }
    rows = []
    for name, (kind, cols) in methods.items():
        for c, sc in data.items():
            y = sc.nil.values
            for folds, groups in [("documents", sc.group.astype(str).values), ("expressions", text[c])]:
                if kind == "none":
                    p = sc[PRIMARY].values
                elif kind == "text":
                    p = grouped_oof(np.arange(len(y)).reshape(-1, 1), y, groups, lambda: _TextModel(text[c]))
                elif kind == "emb":
                    p = grouped_oof(emb[c], y, groups, logistic)
                else:
                    p = grouped_oof(sc[cols].values, y, groups, logistic)
                rows.append(dict(method=name, train=c, test=c, setting=f"within corpus, folds by {folds}", **metrics(y, p)))
        for src, tgt in [("cadec", "psytar"), ("psytar", "cadec")]:
            y = data[tgt].nil.values
            if kind == "none":
                p = data[tgt][PRIMARY].values
            elif kind == "text":
                both = np.concatenate([text[src], text[tgt]])
                m = _TextModel(both).fit(np.arange(len(text[src])).reshape(-1, 1), data[src].nil.values)
                p = m.predict_proba(np.arange(len(text[src]), len(both)).reshape(-1, 1))[:, 1]
            elif kind == "emb":
                p = logistic().fit(emb[src], data[src].nil.values).predict_proba(emb[tgt])[:, 1]
            else:
                p = logistic().fit(data[src][cols].values, data[src].nil.values).predict_proba(data[tgt][cols].values)[:, 1]
            rows.append(dict(method=name, train=src, test=tgt, setting="transfer", **metrics(y, p)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_supervised.csv"), index=False)
    print(out.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------
def operating_points(all_scores):
    rows = []
    for corpus, s in all_scores.items():
        y = scores(corpus)[0].nil.values
        for name, v in s.items():
            order = np.argsort(-v, kind="stable")
            hits = np.cumsum(y[order])
            prec, rec, _ = precision_recall_curve(y, v)
            r = dict(corpus=corpus, method=name, prevalence=y.mean())
            for k in (50, 100, 200, 500):
                r[f"p_at_{k}"] = hits[k - 1] / k
            for target in (0.5, 0.8):
                r[f"precision_at_recall_{int(target * 100)}"] = prec[rec >= target].max()
                r[f"read_for_recall_{int(target * 100)}"] = (np.searchsorted(hits, target * y.sum()) + 1) / len(y)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_operating_points.csv"), index=False)
    print(out.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------
def label_checks(terms):
    names = set(terms.name.map(norm))
    pt_names = set(terms.name[terms.level == "PT"].map(norm))
    rows = []
    for corpus in CORPORA:
        sc, _ = scores(corpus)
        y = sc.nil.values
        verb = sc.text.map(norm).isin(names).values
        subsets = {"all mentions": np.ones(len(y), bool),
                   "verbatim-term positives removed": ~(verb & (y == 1))}
        for sub, m in subsets.items():
            for key in ["string", PRIMARY, "hybrid"] + [k for e in ENCODERS for k in (f"{e}|max|PT", f"{e}|max|PT+LLT", f"{e}|top5|PT+LLT")]:
                rows.append(dict(corpus=corpus, subset=sub, level="mention", method=key, n=int(m.sum()),
                                 positives=int(y[m].sum()), **metrics(y[m], sc[key].values[m])))
        # Distinct-expression level: one row per normalised expression; label = majority of its mentions.
        d = sc.assign(n=sc.text.map(norm)).groupby("n").agg(nil=("nil", "mean"), **{
            k: (k, "mean") for k in ["string", PRIMARY, "hybrid"]})
        yy = (d.nil >= 0.5).astype(int).values
        for key in ["string", PRIMARY, "hybrid"]:
            rows.append(dict(corpus=corpus, subset="all", level="distinct expression", method=key, n=len(d),
                             positives=int(yy.sum()), **metrics(yy, d[key].values)))
        rows.append(dict(corpus=corpus, subset="verbatim PT names among positives", level="count",
                         method="-", n=int((verb & (y == 1)).sum()),
                         positives=int((sc.text.map(norm).isin(pt_names).values & (y == 1)).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_label_checks.csv"), index=False)
    print(out.round(3).to_string(index=False))


def main():
    terms = pd.read_csv(os.path.join(OUT_DIR, "meddra_terms.csv"))
    _, llt2pt = load_meddra()
    rng = np.random.default_rng(SEED)
    print("== A. length"); length_analysis()
    print("== B. score variants"); score_variants(terms, llt2pt)
    print("== C. baselines"); s = baselines(terms, rng)
    print("== D. hybrid"); hybrid_sensitivity()
    print("== D2. supervised versus label-free"); supervised_versus_label_free()
    print("== E. operating points"); operating_points(s)
    print("== F. labels"); label_checks(terms)


if __name__ == "__main__":
    main()
