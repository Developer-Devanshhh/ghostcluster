"""Step 7 - additional analyses requested in review (Sections 5.3-5.7 and Supplementary Information).

A. Alternative out-of-distribution scores: density-normalised gap (distance relative to the local
   density of the terminology) and Mahalanobis distance to the terminology embeddings.
B. BM25 parameter grid and a query-length-normalised BM25.
C. Supervision: a hybrid with the same features as the consensus; transfer of a low-dimensional model
   (length + consensus); same-terminology transfer between drugs within CADEC (atorvastatin and
   diclofenac); learning curves against the number of labelled mentions.
D. Batch dependence of the rank-based consensus and a reference-batch version for deployment.
E. Spelling normalisation against the MedDRA vocabulary before scoring.
F. Uncertainty of operating points; terminology-density strata with counts and intervals.
G. Ward clustering on the concatenated embeddings of all encoders.
H-L. Mention, distinct-expression and length-adjusted results; intervals for transfer; encoder stability;
   rule-based error categories; cross-encoder and fine-tuned normaliser baselines.

Outputs (outputs/): results_ood_scores.csv, results_bm25.csv, results_transfer_extra.csv,
results_learning_curve.csv, results_batch.csv, results_spelling.csv, results_operating_ci.csv,
results_density.csv, results_triage_concat.csv, results_primary_levels.csv, results_transfer_ci.csv,
results_encoder_stability.csv, results_model_baselines.csv, results_error_analysis.csv
"""
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ghostcluster.clustering import ghost_clusters, ward_k, ward_labels
from ghostcluster.config import ENCODERS, OUT_DIR, PRIMARY, SEED
from ghostcluster.data import load_meddra, norm
from ghostcluster.encoders import load_encoder
from ghostcluster.scoring import consensus_gap, top_k_similarities
from ghostcluster.stats import _resample

_spec = importlib.util.spec_from_file_location("sens", os.path.join(HERE, "05_sensitivity.py"))
sens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sens)
scores, metrics, grouped_oof, logistic, TextModel = sens.scores, sens.metrics, sens.grouped_oof, sens.logistic, sens._TextModel

CORPORA = ["cadec", "psytar"]
TOP5 = [f"{e}|top5|PT+LLT" for e in ENCODERS]


def load(name):
    return np.load(os.path.join(OUT_DIR, name))


def text_gaps(corpus, e, k=5):
    top, arg = top_k_similarities(load(f"emb_{e}_{corpus}.npy"), load(f"emb_{e}_meddra.npy"), k=k)
    return 1 - top.mean(1), arg


# ---------------------------------------------------------------------------------------------
def term_density(e):
    """For every terminology term: 1 - mean similarity to its five nearest other terms (cached)."""
    path = os.path.join(OUT_DIR, f"term_density_{e}.npy")
    if os.path.exists(path):
        return np.load(path)
    M = load(f"emb_{e}_meddra.npy")
    top, _ = top_k_similarities(M, M, k=6)
    r = 1 - top[:, 1:].mean(1)        # column 0 is the term itself
    np.save(path, r)
    return r


def ood_scores():
    rows = []
    for corpus in CORPORA:
        sc, _ = scores(corpus)
        y, ix = sc.nil.values, sc.ti.values
        dn, mh = [], []
        for e in ENCODERS:
            g, arg = text_gaps(corpus, e)
            r = term_density(e)
            dn.append(g / np.maximum(r[arg].mean(1), 1e-6))
            M = load(f"emb_{e}_meddra.npy")
            lw = LedoitWolf().fit(M[np.random.default_rng(SEED).choice(len(M), 20000, replace=False)])
            mh.append(lw.mahalanobis(load(f"emb_{e}_{corpus}.npy")))
            for name, s in [("density-normalised gap", dn[-1]), ("Mahalanobis distance", mh[-1])]:
                rows.append(dict(corpus=corpus, score=name, encoder=e, **metrics(y, s[ix])))
        for name, parts in [("density-normalised gap", dn), ("Mahalanobis distance", mh)]:
            rows.append(dict(corpus=corpus, score=name, encoder=PRIMARY, **metrics(y, consensus_gap(parts)[ix])))
        rows.append(dict(corpus=corpus, score="top-5 gap", encoder=PRIMARY, **metrics(y, sc[PRIMARY].values)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_ood_scores.csv"), index=False)
    print(out.pivot_table(index=["score", "encoder"], columns="corpus", values=["auroc", "auprc"]).round(3).to_string())


# ---------------------------------------------------------------------------------------------
def bm25_matrix(names, k1, b):
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    D = cv.fit_transform(names).tocsr().astype(np.float64)
    N, dl = D.shape[0], np.asarray(D.sum(1)).ravel()
    df = np.bincount(D.indices, minlength=D.shape[1])
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    D = D.tocoo()
    w = D.data * (k1 + 1) / (D.data + k1 * (1 - b + b * dl[D.row] / dl.mean())) * idf[D.col]
    from scipy.sparse import csr_matrix
    return cv, csr_matrix((w, (D.row, D.col)), shape=D.shape), idf


def bm25():
    terms = pd.read_csv(os.path.join(OUT_DIR, "meddra_terms.csv"))
    names = [norm(n) for n in terms.name]
    rows = []
    for k1 in (0.5, 1.2, 2.0):
        for b in (0.25, 0.75, 1.0):
            cv, W, idf = bm25_matrix(names, k1, b)
            for corpus in CORPORA:
                sc, texts = scores(corpus)
                y, ix = sc.nil.values, sc.ti.values
                Q = (cv.transform([norm(t) for t in texts]) > 0).astype(np.float64)
                best = np.zeros(len(texts))
                for i in range(0, len(texts), 500):
                    best[i:i + 500] = (Q[i:i + 500] @ W.T).max(1).toarray().ravel()
                # Upper bound of the score for this query: every query term matched with maximal weight.
                bound = np.asarray(Q @ (idf * (k1 + 1))).ravel()
                rows.append(dict(corpus=corpus, k1=k1, b=b, variant="raw", **metrics(y, -best[ix])))
                rows.append(dict(corpus=corpus, k1=k1, b=b, variant="normalised by query bound",
                                 **metrics(y, -(best / np.maximum(bound, 1e-9))[ix])))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_bm25.csv"), index=False)
    print(out.pivot_table(index=["variant", "k1", "b"], columns="corpus", values="auroc").round(3).to_string())


# ---------------------------------------------------------------------------------------------
def supervision():
    data = {c: scores(c)[0] for c in CORPORA}
    text = {c: np.array(d.text.map(norm)) for c, d in data.items()}
    emb = {c: load(f"emb_biolord_{c}.npy")[d.ti.values] for c, d in data.items()}
    feats = {
        "Hybrid, consensus features (length, string, four top-5 gaps)": ["length", "string"] + TOP5,
        "Length + consensus": ["length", PRIMARY],
    }
    rows = []
    # (a) cross-corpus transfer of low-dimensional models, and within-corpus reference values
    for name, cols in feats.items():
        for c, sc in data.items():
            p = grouped_oof(sc[cols].values, sc.nil.values, sc.group.astype(str).values, logistic)
            rows.append(dict(setting="within corpus", train=c, test=c, method=name, **metrics(sc.nil.values, p)))
        for src, tgt in [("cadec", "psytar"), ("psytar", "cadec")]:
            m = logistic().fit(data[src][cols].values, data[src].nil.values)
            p = m.predict_proba(data[tgt][cols].values)[:, 1]
            rows.append(dict(setting="across corpora", train=src, test=tgt, method=name, **metrics(data[tgt].nil.values, p)))
    # (b) same terminology, different drug: CADEC atorvastatin (Lipitor) versus diclofenac posts
    sc = data["cadec"]
    statin = sc.group.str.startswith("LIPITOR").values
    for src_mask, tgt_mask, src, tgt in [(statin, ~statin, "atorvastatin", "diclofenac"),
                                         (~statin, statin, "diclofenac", "atorvastatin")]:
        y_s, y_t = sc.nil.values[src_mask], sc.nil.values[tgt_mask]
        preds = {}
        both = np.concatenate([text["cadec"][src_mask], text["cadec"][tgt_mask]])
        tm = TextModel(both).fit(np.arange(src_mask.sum()).reshape(-1, 1), y_s)
        preds["Supervised TF-IDF classifier"] = tm.predict_proba(np.arange(src_mask.sum(), len(both)).reshape(-1, 1))[:, 1]
        preds["Supervised embedding classifier (BioLORD-2023)"] = logistic().fit(emb["cadec"][src_mask], y_s).predict_proba(emb["cadec"][tgt_mask])[:, 1]
        hyb = ["length", "string"] + [f"{e}|max|PT+LLT" for e in ["spubmedbert", "sapbert"]]
        preds["Hybrid (supervised)"] = logistic().fit(sc[hyb].values[src_mask], y_s).predict_proba(sc[hyb].values[tgt_mask])[:, 1]
        for name, cols in feats.items():
            preds[name] = logistic().fit(sc[cols].values[src_mask], y_s).predict_proba(sc[cols].values[tgt_mask])[:, 1]
        preds["Consensus gap (label-free)"] = sc[PRIMARY].values[tgt_mask]
        for name, p in preds.items():
            rows.append(dict(setting="across drugs (CADEC)", train=src, test=tgt, method=name, prevalence=y_t.mean(),
                             **metrics(y_t, p)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_transfer_extra.csv"), index=False)
    print(out.round(3).to_string(index=False))

    # (c) learning curves: train on n labelled mentions from four folds, test on the fifth
    from sklearn.model_selection import GroupKFold
    lc = []
    for c, sc in data.items():
        y, g = sc.nil.values, sc.group.astype(str).values
        for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(sc, y, g)):
            for n in (50, 100, 250, 500, 1000, 2000, len(train)):
                for rep in range(5 if n < len(train) else 1):
                    rng = np.random.default_rng(1000 * fold + rep)
                    sub = train if n >= len(train) else rng.choice(train, n, replace=False)
                    if y[sub].sum() < 2:
                        continue
                    tm = TextModel(text[c]).fit(sub.reshape(-1, 1), y[sub])
                    p_tf = tm.predict_proba(test.reshape(-1, 1))[:, 1]
                    p_lc = logistic().fit(sc[["length", PRIMARY]].values[sub], y[sub]).predict_proba(sc[["length", PRIMARY]].values[test])[:, 1]
                    for name, p in [("Supervised TF-IDF classifier", p_tf), ("Length + consensus", p_lc),
                                    ("Consensus gap (label-free)", sc[PRIMARY].values[test])]:
                        lc.append(dict(corpus=c, fold=fold, n=min(n, len(train)), rep=rep, method=name,
                                       auprc=average_precision_score(y[test], p)))
    lc = pd.DataFrame(lc)
    lc["n_label"] = lc.groupby(["corpus", "fold"]).n.transform(lambda s: s.where(s < s.max(), -1))
    lc.to_csv(os.path.join(OUT_DIR, "results_learning_curve.csv"), index=False)
    print(lc.groupby(["corpus", "method", "n"]).auprc.mean().unstack().round(3).to_string())


# ---------------------------------------------------------------------------------------------
def batches():
    """Rank consensus within random document subsets, and against a fixed reference batch (SMM4H)."""
    ref = {e: np.sort(text_gaps("smm4h", e)[0]) for e in ENCODERS}
    rows = []
    for corpus in CORPORA:
        sc, texts = scores(corpus)
        y, ix, g = sc.nil.values, sc.ti.values, sc.group.astype(str).values
        gaps = {e: text_gaps(corpus, e)[0] for e in ENCODERS}
        ref_cons = np.mean([np.searchsorted(ref[e], gaps[e], side="right") / len(ref[e]) for e in ENCODERS], 0)
        rows.append(dict(corpus=corpus, variant="reference batch (SMM4H-2017)", fraction=1.0,
                         spearman_with_full=spearmanr(ref_cons[ix], sc[PRIMARY].values)[0], **metrics(y, ref_cons[ix])))
        rows.append(dict(corpus=corpus, variant="within batch (full corpus)", fraction=1.0, spearman_with_full=1.0,
                         **metrics(y, sc[PRIMARY].values)))
        docs = np.unique(g)
        for frac in (0.05, 0.1, 0.25, 0.5):
            for rep in range(20):
                keep = np.isin(g, np.random.default_rng(rep).choice(docs, int(frac * len(docs)), replace=False))
                if y[keep].sum() < 3:
                    continue
                t_ix = np.unique(ix[keep])
                local = consensus_gap([gaps[e][t_ix] for e in ENCODERS])
                pos = {t: i for i, t in enumerate(t_ix)}
                s_local = local[[pos[t] for t in ix[keep]]]
                rows.append(dict(corpus=corpus, variant="within batch (document subset)", fraction=frac, rep=rep,
                                 spearman_with_full=spearmanr(s_local, sc[PRIMARY].values[keep])[0],
                                 auprc_full_score=average_precision_score(y[keep], sc[PRIMARY].values[keep]),
                                 **metrics(y[keep], s_local)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_batch.csv"), index=False)
    print(out.groupby(["corpus", "variant", "fraction"])[["spearman_with_full", "auroc", "auprc"]].agg(["mean", "min"]).round(3).to_string())


# ---------------------------------------------------------------------------------------------
def spelling():
    """Correct out-of-vocabulary words to the closest MedDRA word (edit similarity >= 85, same first letter)."""
    terms = pd.read_csv(os.path.join(OUT_DIR, "meddra_terms.csv"))
    vocab = sorted({w for n in terms.name for w in re.findall(r"[a-z]+", str(n).lower())})
    by_letter = {}
    for w in vocab:
        by_letter.setdefault(w[0], []).append(w)
    vset = set(vocab)
    cache = {}

    def fix_word(w):
        if len(w) < 5 or w in vset:
            return w
        if w not in cache:
            m = process.extractOne(w, by_letter.get(w[0], []), scorer=fuzz.ratio, score_cutoff=85)
            cache[w] = m[0] if m else w
        return cache[w]

    rows, examples = [], []
    encoders = {}
    for corpus in CORPORA:
        sc, texts = scores(corpus)
        y, ix = sc.nil.values, sc.ti.values
        fixed = [" ".join(fix_word(w) for w in re.findall(r"[a-z]+|[^a-z\s]+", t.lower())) for t in texts]
        changed = np.array([norm(a) != norm(b) for a, b in zip(texts, fixed)])
        examples += [dict(corpus=corpus, original=a, corrected=b) for a, b, ch in zip(texts, fixed, changed) if ch][:15]
        gaps_new = []
        for e, (model_id, pooling) in ENCODERS.items():
            if e not in encoders:
                encoders[e] = load_encoder(model_id, pooling)
            g, _ = text_gaps(corpus, e)
            if changed.any():
                Q = encoders[e]([fixed[i] for i in np.where(changed)[0]])
                top, _ = top_k_similarities(Q, load(f"emb_{e}_meddra.npy"), k=5)
                g = g.copy()
                g[changed] = 1 - top.mean(1)
            gaps_new.append(g)
        s_new = consensus_gap(gaps_new)[ix]
        for name, s in [("Consensus gap", sc[PRIMARY].values), ("Consensus gap after spelling normalisation", s_new)]:
            order = np.argsort(-s, kind="stable")
            rows.append(dict(corpus=corpus, score=name, changed_expressions=int(changed.sum()),
                             changed_mentions=int(changed[ix].sum()), p_at_100=y[order[:100]].mean(), **metrics(y, s)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_spelling.csv"), index=False)
    pd.DataFrame(examples).to_csv(os.path.join(OUT_DIR, "spelling_examples.csv"), index=False)
    print(out.round(3).to_string(index=False))
    print(pd.DataFrame(examples).head(15).to_string(index=False))


# ---------------------------------------------------------------------------------------------
def operating_ci_and_density():
    rng = np.random.default_rng(SEED)
    rows = []
    for corpus in CORPORA:
        sc, _ = scores(corpus)
        y, g = sc.nil.values, sc.group.astype(str).values
        for name, key in [("String match", "string"), ("Consensus gap", PRIMARY), ("Hybrid", "hybrid"), ("Dense normaliser", "sapbert|max|PT+LLT")]:
            s = sc[key].values

            def ops(ix):
                yy, ss = y[ix], s[ix]
                o = np.argsort(-ss, kind="stable")
                hits = np.cumsum(yy[o])
                return (yy[o[:100]].mean(), (np.searchsorted(hits, 0.5 * yy.sum()) + 1) / len(yy),
                        (np.searchsorted(hits, 0.8 * yy.sum()) + 1) / len(yy))
            point = ops(np.arange(len(y)))
            draws = _resample(g, rng)
            boot = np.array([ops(next(draws)) for _ in range(1000)])
            lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
            rows.append(dict(corpus=corpus, method=name, p_at_100=point[0], p_at_100_lo=lo[0], p_at_100_hi=hi[0],
                             read_r50=point[1], read_r50_lo=lo[1], read_r50_hi=hi[1],
                             read_r80=point[2], read_r80_lo=lo[2], read_r80_hi=hi[2]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "results_operating_ci.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    _, llt2pt = load_meddra()
    terms = pd.read_csv(os.path.join(OUT_DIR, "meddra_terms.csv"))
    parent_of = dict(zip(llt2pt.llt, llt2pt.pt))
    parent = np.array([c if lvl == "PT" else parent_of.get(c, c) for c, lvl in zip(terms.code, terms.level)])
    n_llt = pd.Series(parent[(terms.level == "LLT").values]).value_counts()
    dens = []
    for corpus in CORPORA:
        sc, _ = scores(corpus)
        y, g = sc.nil.values, sc.group.astype(str).values
        _, arg = text_gaps(corpus, "biolord")
        d = pd.Series(parent[arg[:, 0]]).map(n_llt).fillna(0).values[sc.ti.values]
        tert = pd.qcut(d, 3, labels=["low", "middle", "high"], duplicates="drop")
        for t in tert.categories:
            m = np.asarray(tert == t)
            from ghostcluster.stats import bootstrap_metrics
            r = bootstrap_metrics(y[m], sc[PRIMARY].values[m], g[m], np.random.default_rng(SEED))
            dens.append(dict(corpus=corpus, tertile=t, llt_range=f"{int(d[m].min())}-{int(d[m].max())}", mentions=int(m.sum()),
                             positives=int(y[m].sum()), prevalence=y[m].mean(), **r))
    pd.DataFrame(dens).to_csv(os.path.join(OUT_DIR, "results_density.csv"), index=False)
    print(pd.DataFrame(dens).round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------
def concatenated_clustering():
    tri = importlib.util.spec_from_file_location("rob", os.path.join(HERE, "04_robustness.py"))
    rob = importlib.util.module_from_spec(tri)
    tri.loader.exec_module(rob)
    rows = []
    for corpus in CORPORA:
        sc, texts = scores(corpus)
        gap = sc.groupby("text")[PRIMARY].mean().reindex(texts).values
        E = np.hstack([load(f"emb_{e}_{corpus}.npy") for e in ENCODERS])
        for m in (10, 20, 50):
            lab = ward_labels(E, ward_k(len(texts), m))
            r, _ = rob.triage(f"Ward on concatenated embeddings, {m} per cluster", corpus, texts, lab, gap, sc)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_triage_concat.csv"), index=False)
    print(out[["corpus", "run", "ghost_clusters", "expressions", "covered", "enrichment", "captured",
               "captured_by_ranking", "ghosts_majority_unmappable"]].round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------
# Additions in the second revision round
# ---------------------------------------------------------------------------------------------
def length_residual(s, length):
    """Remove the least-squares fit of a score on log length (no labels; fitted on the whole batch)."""
    L = np.column_stack([np.ones(len(s)), np.log(length)])
    return s - L @ np.linalg.lstsq(L, s, rcond=None)[0]


def primary_levels():
    """Mention-level, distinct-expression-level and length-adjusted results with bootstrap intervals."""
    from ghostcluster.stats import bootstrap_metrics
    rows = []
    for corpus in CORPORA:
        sc, _ = scores(corpus)
        y, g = sc.nil.values, sc.group.astype(str).values
        sc["length+string"] = grouped_oof(sc[["length", "string"]].values, y, g, logistic)
        methods = {"Length (words)": "length", "String match": "string", "Length + string (supervised)": "length+string",
                   "Dense normaliser confidence": "sapbert|max|PT+LLT", "Hybrid (supervised)": "hybrid",
                   "Consensus gap (label-free)": PRIMARY}
        sc["n"] = sc.text.map(norm)
        d = sc.groupby("n").agg(nil=("nil", "mean"), length=("length", "first"),
                                **{k: (k, "mean") for k in methods.values() if k != "length"})
        yd = (d.nil >= 0.5).astype(int).values
        for name, key in methods.items():
            r = dict(corpus=corpus, method=name)
            m = bootstrap_metrics(y, sc[key].values, g, np.random.default_rng(SEED))
            r.update({f"mention_{k}": v for k, v in m.items()})
            m = bootstrap_metrics(yd, d[key].values, d.index.values, np.random.default_rng(SEED))
            r.update({f"distinct_{k}": v for k, v in m.items()})
            if key not in ("length", "length+string", "hybrid"):
                m = bootstrap_metrics(y, length_residual(sc[key].values, sc.length.values), g, np.random.default_rng(SEED))
                r.update({f"lenadj_{k}": v for k, v in m.items()})
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_primary_levels.csv"), index=False)
    print(out[["corpus", "method", "mention_auprc", "distinct_auprc", "lenadj_auprc"]].round(3).to_string(index=False))


def transfer_intervals():
    """Bootstrap intervals (test documents) and paired differences with the consensus for every transfer cell."""
    from ghostcluster.stats import bootstrap_metrics, paired_auprc_difference
    data = {c: scores(c)[0] for c in CORPORA}
    text = {c: np.array(d.text.map(norm)) for c, d in data.items()}
    emb = {c: load(f"emb_biolord_{c}.npy")[d.ti.values] for c, d in data.items()}
    hyb = ["length", "string"] + [f"{e}|max|PT+LLT" for e in ["spubmedbert", "sapbert"]]
    hyb4 = ["length", "string"] + TOP5

    def fit_predict(kind, Xs, ys, Xt, ts=None, tt=None):
        if kind == "tfidf":
            both = np.concatenate([ts, tt])
            m = TextModel(both).fit(np.arange(len(ts)).reshape(-1, 1), ys)
            return m.predict_proba(np.arange(len(ts), len(both)).reshape(-1, 1))[:, 1]
        return logistic().fit(Xs, ys).predict_proba(Xt)[:, 1]

    cells = []
    sc = data["cadec"]
    statin = sc.group.str.startswith("LIPITOR").values
    for a, b, an, bn in [(statin, ~statin, "atorvastatin", "diclofenac"), (~statin, statin, "diclofenac", "atorvastatin")]:
        cells.append(("across drugs (CADEC)", an, bn, sc[a], sc[b], text["cadec"][a], text["cadec"][b], emb["cadec"][a], emb["cadec"][b]))
    for s_, t_ in [("cadec", "psytar"), ("psytar", "cadec")]:
        cells.append(("across corpora", s_, t_, data[s_], data[t_], text[s_], text[t_], emb[s_], emb[t_]))
    rows = []
    for setting, sn, tn, S, T, ts, tt, es, et in cells:
        y, g = T.nil.values, T.group.astype(str).values
        preds = {
            "Supervised TF-IDF classifier": fit_predict("tfidf", None, S.nil.values, None, ts, tt),
            "Supervised embedding classifier (BioLORD-2023)": fit_predict("lr", es, S.nil.values, et),
            "Hybrid (two encoders, top-1)": fit_predict("lr", S[hyb].values, S.nil.values, T[hyb].values),
            "Hybrid (four encoders, top-5)": fit_predict("lr", S[hyb4].values, S.nil.values, T[hyb4].values),
            "Length + consensus (supervised)": fit_predict("lr", S[["length", PRIMARY]].values, S.nil.values, T[["length", PRIMARY]].values),
            "Consensus gap (label-free)": T[PRIMARY].values,
        }
        for name, p in preds.items():
            m = bootstrap_metrics(y, p, g, np.random.default_rng(SEED))
            r = dict(setting=setting, train=sn, test=tn, method=name, prevalence=y.mean(), positives=int(y.sum()), **m)
            if name != "Consensus gap (label-free)":
                d, lo, hi, pv = paired_auprc_difference(y, preds["Consensus gap (label-free)"], p, g, np.random.default_rng(SEED))
                r.update(consensus_minus=d, cm_lo=lo, cm_hi=hi, cm_p=pv)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_transfer_ci.csv"), index=False)
    print(out[["setting", "train", "test", "method", "auprc", "auprc_lo", "auprc_hi", "consensus_minus", "cm_lo", "cm_hi"]].round(3).to_string(index=False))


def encoder_stability():
    from scipy.stats import rankdata
    rows = []
    for corpus in CORPORA:
        sc, texts = scores(corpus)
        y, ix = sc.nil.values, sc.ti.values
        gaps = {e: text_gaps(corpus, e)[0] for e in ENCODERS}
        full = consensus_gap(list(gaps.values()))
        top_full = set(np.argsort(-full)[:200])
        rows.append(dict(corpus=corpus, variant="all four encoders (rank average)", spearman=1.0, top200_overlap=1.0, **metrics(y, full[ix])))
        z = np.mean([(v - v.mean()) / v.std() for v in gaps.values()], 0)
        rows.append(dict(corpus=corpus, variant="all four encoders (z-score average)", spearman=spearmanr(z, full)[0],
                         top200_overlap=len(top_full & set(np.argsort(-z)[:200])) / 200, **metrics(y, z[ix])))
        for e in ENCODERS:
            rest = consensus_gap([gaps[k] for k in ENCODERS if k != e])
            rows.append(dict(corpus=corpus, variant=f"without {e}", spearman=spearmanr(rest, full)[0],
                             top200_overlap=len(top_full & set(np.argsort(-rest)[:200])) / 200, **metrics(y, rest[ix])))
            rows.append(dict(corpus=corpus, variant=f"{e} alone", spearman=spearmanr(gaps[e], full)[0],
                             top200_overlap=len(top_full & set(np.argsort(-gaps[e])[:200])) / 200, **metrics(y, gaps[e][ix])))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_encoder_stability.csv"), index=False)
    print(out.round(3).to_string(index=False))


def _encode_texts(model, texts, batch=256):
    return model.encode(list(texts), batch_size=batch, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float32)


def model_baselines():
    """Cross-encoder re-ranker confidence and a normaliser fine-tuned on SMM4H-2017 MedDRA mappings."""
    from sentence_transformers import CrossEncoder, SentenceTransformer, models
    from ghostcluster.data import load_smm4h
    terms = pd.read_csv(os.path.join(OUT_DIR, "meddra_terms.csv"))
    names = terms.name.astype(str).values
    rows = []

    # (a) cross-encoder re-ranking of the 20 nearest BioLORD-2023 candidates
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=64, device="cpu")
    ce_gap = {}
    for corpus in CORPORA:
        sc, texts = scores(corpus)
        _, arg = top_k_similarities(load(f"emb_biolord_{corpus}.npy"), load("emb_biolord_meddra.npy"), k=20)
        pairs = [(t, names[j]) for t, a in zip(texts, arg) for j in a]
        s = np.asarray(ce.predict(pairs, batch_size=256, show_progress_bar=False)).reshape(len(texts), 20)
        ce_gap[corpus] = -s.max(1)
        rows.append(dict(corpus=corpus, method="Cross-encoder re-ranker confidence (MS MARCO MiniLM, top-20 BioLORD candidates)",
                         **metrics(sc.nil.values, ce_gap[corpus][sc.ti.values])))
    # (b) SapBERT fine-tuned on SMM4H-2017 mention -> gold term name pairs (normalisation supervision, no no-concept labels)
    smm = load_smm4h()
    name_of = dict(zip(terms.code, terms.name.astype(str)))
    train = smm.dropna(subset=["gold"]).assign(term=lambda d: d.gold.map(name_of)).dropna(subset=["term"])
    train = train.drop_duplicates(["text", "term"])
    word = models.Transformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", max_seq_length=32)
    model = SentenceTransformer(modules=[word, models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")], device="cpu")
    for label, fine_tune in [("SapBERT, not fine-tuned (same pipeline)", False), ("SapBERT fine-tuned on SMM4H-2017 normalisation", True)]:
        if fine_tune:
            # One epoch of in-batch-negatives contrastive training (multiple-negatives ranking loss, scale 20).
            import torch
            torch.manual_seed(SEED)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
            order = np.random.default_rng(SEED).permutation(len(train))
            a_txt, b_txt = train.text.values[order], train.term.values[order]
            model.train()
            for i in range(0, len(order), 32):
                fa = {k: v for k, v in model.tokenize(list(a_txt[i:i + 32])).items()}
                fb = {k: v for k, v in model.tokenize(list(b_txt[i:i + 32])).items()}
                ea = torch.nn.functional.normalize(model(fa)["sentence_embedding"], dim=-1)
                eb = torch.nn.functional.normalize(model(fb)["sentence_embedding"], dim=-1)
                loss = torch.nn.functional.cross_entropy(20 * ea @ eb.T, torch.arange(len(ea)))
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
        M = _encode_texts(model, names)
        for corpus in CORPORA:
            sc, texts = scores(corpus)
            top, _ = top_k_similarities(_encode_texts(model, texts), M, k=1)
            rows.append(dict(corpus=corpus, method=f"Normaliser confidence: {label}", training_pairs=len(train) if fine_tune else 0,
                             **metrics(sc.nil.values, (1 - top[:, 0])[sc.ti.values])))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_model_baselines.csv"), index=False)
    print(out.round(3).to_string(index=False))


CONTEXT = re.compile(r"\b(er|e\.r|hospital\w*|emergency|admitted|doctor|dr|physician|surgery|test\w*|blood work|prescri\w*|insurance|pharmac\w*|dose|dosage|mg)\b", re.I)


def error_analysis():
    """Rule-based categories for the 100 highest-ranked coded mentions and the 100 lowest-ranked no-concept mentions."""
    terms = pd.read_csv(os.path.join(OUT_DIR, "meddra_terms.csv"))
    vocab = {w for n in terms.name for w in re.findall(r"[a-z]+", str(n).lower())}
    names = list(dict.fromkeys(norm(n) for n in terms.name))
    rows, examples = [], []
    for corpus in CORPORA:
        sc, _ = scores(corpus)
        d = sc.assign(n=sc.text.map(norm)).drop_duplicates("n")
        fp = d[d.nil == 0].sort_values(PRIMARY, ascending=False).head(100)
        fn = d[d.nil == 1].sort_values(PRIMARY).head(100)
        for kind, part in [("false positive (top-ranked, coded)", fp), ("false negative (lowest-ranked, no concept)", fn)]:
            for _, r in part.iterrows():
                words = re.findall(r"[a-z]+", r.n)
                best = process.extractOne(r.n, names, scorer=fuzz.token_sort_ratio)[1]
                if CONTEXT.search(r.n):
                    cat = "health-care use or context"
                elif any(len(w) >= 5 and w not in vocab and process.extractOne(w, vocab, scorer=fuzz.ratio, score_cutoff=85) for w in words):
                    cat = "misspelling or non-standard word form"
                elif len(words) >= 6:
                    cat = "long description (6+ words)"
                elif best >= 90:
                    cat = "near-verbatim MedDRA term"
                elif len(words) <= 2:
                    cat = "short expression (1-2 words)"
                else:
                    cat = "other"
                rows.append(dict(corpus=corpus, kind=kind, category=cat))
                examples.append(dict(corpus=corpus, kind=kind, category=cat, text=r.text))
    out = pd.DataFrame(rows).value_counts().rename("count").reset_index()
    out.to_csv(os.path.join(OUT_DIR, "results_error_analysis.csv"), index=False)
    pd.DataFrame(examples).to_csv(os.path.join(OUT_DIR, "error_examples.csv"), index=False)
    print(out.sort_values(["corpus", "kind", "count"], ascending=[True, True, False]).to_string(index=False))


def within_intervals():
    """Within-corpus, document-grouped out-of-fold predictions with bootstrap intervals and paired differences."""
    from ghostcluster.stats import bootstrap_metrics, paired_auprc_difference
    hyb = ["length", "string"] + [f"{e}|max|PT+LLT" for e in ["spubmedbert", "sapbert"]]
    hyb4 = ["length", "string"] + TOP5
    rows = []
    for c in CORPORA:
        sc, _ = scores(c)
        y, g = sc.nil.values, sc.group.astype(str).values
        tx = np.array(sc.text.map(norm))
        preds = {
            "Supervised TF-IDF classifier": grouped_oof(np.arange(len(y)).reshape(-1, 1), y, g, lambda: TextModel(tx)),
            "Supervised embedding classifier (BioLORD-2023)": grouped_oof(load(f"emb_biolord_{c}.npy")[sc.ti.values], y, g, logistic),
            "Hybrid (two encoders, top-1)": sc.hybrid.values,
            "Hybrid (four encoders, top-5)": grouped_oof(sc[hyb4].values, y, g, logistic),
            "Length + consensus (supervised)": grouped_oof(sc[["length", PRIMARY]].values, y, g, logistic),
            "Consensus gap (label-free)": sc[PRIMARY].values,
        }
        for name, p in preds.items():
            r = dict(setting="within corpus", train=c, test=c, method=name, prevalence=y.mean(), positives=int(y.sum()),
                     **bootstrap_metrics(y, p, g, np.random.default_rng(SEED)))
            if name != "Consensus gap (label-free)":
                d, lo, hi, pv = paired_auprc_difference(y, preds["Consensus gap (label-free)"], p, g, np.random.default_rng(SEED))
                r.update(consensus_minus=d, cm_lo=lo, cm_hi=hi, cm_p=pv)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "results_within_ci.csv"), index=False)
    print(out[["test", "method", "auprc", "auprc_lo", "auprc_hi", "consensus_minus", "cm_lo", "cm_hi"]].round(3).to_string(index=False))


def round2():
    print("== H. primary levels"); primary_levels()
    print("== I. transfer intervals"); transfer_intervals(); within_intervals()
    print("== J. encoder stability"); encoder_stability()
    print("== K. error analysis"); error_analysis()
    print("== L. model baselines"); model_baselines()


def main():
    print("== A. OOD scores"); ood_scores()
    print("== B. BM25"); bm25()
    print("== C. supervision"); supervision()
    print("== D. batches"); batches()
    print("== F. operating CIs and density"); operating_ci_and_density()
    print("== G. concatenated clustering"); concatenated_clustering()
    print("== E. spelling"); spelling()
    round2()


if __name__ == "__main__":
    main()
