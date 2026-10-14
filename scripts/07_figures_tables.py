"""Step 7 - figures (PDF) and LaTeX tables of the manuscript and its Supplementary Information.

Outputs: outputs/figures/*.pdf and outputs/tables/*.tex (included verbatim by the manuscript).
Colours are a colour-vision-deficiency-checked categorical palette; line styles differ as well,
so methods are never distinguished by colour alone.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ghostcluster.config import ENCODER_NAMES, ENCODERS, FIG_DIR, OUT_DIR, PRIMARY, ROOT, TAB_DIR, WARD_MEAN_SIZE

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42})
CORPORA = {"cadec": "CADEC (MedDRA CONCEPT_LESS)", "psytar": "PsyTAR (no SNOMED CT code)"}
NAME = {"cadec": "CADEC", "psytar": "PsyTAR", "smm4h": "SMM4H-2017"}
CURVES = {
    "length": ("Length (words)", "#2a78d6", (0, (1, 1.5))),
    "string": ("String match", "#eb6834", (0, (4, 2))),
    "sapbert|max|PT+LLT": ("Dense normaliser confidence", "#1baf7a", (0, (6, 1.5, 1, 1.5))),
    "hybrid": ("Hybrid (supervised)", "#eda100", (0, (8, 2))),
    PRIMARY: ("Consensus gap (label-free)", "#c2185b", "solid"),
}


def label(key):
    fixed = {"length": "Length (words)", "string": "String match (token-sort ratio)",
             "hybrid": "Hybrid (supervised)", PRIMARY: "Consensus gap (label-free)"}
    if key in fixed:
        return fixed[key]
    enc, kind, ref = key.split("|")
    return f"{ENCODER_NAMES[enc]}, {'top-5 mean' if kind == 'top5' else 'top-1'}, {'PT only' if ref == 'PT' else 'PT+LLT'}"


def read(name):
    return pd.read_csv(os.path.join(OUT_DIR, name))


def write(name, text):
    with open(os.path.join(TAB_DIR, name), "w", encoding="utf-8") as fh:
        fh.write(text)


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def ci(r, m="auprc"):
    return f"{getattr(r, m):.3f} [{getattr(r, m + '_lo'):.3f}, {getattr(r, m + '_hi'):.3f}]"


def table(caption, label_, spec, header, body, star=False, size=r"\small", extra=""):
    env = "table*" if star else "table"
    return (f"\\begin{{{env}}}[{'t' if star else 'htbp!'}]\n\\caption{{{caption}}}\n\\label{{{label_}}}\n"
            f"\\centering{size}\n{extra}\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            + "\n".join(body) + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def bold_row(cells):
    return " & ".join(r"\textbf{" + c + "}" if c.strip() else c for c in cells.split(" & "))


# ---------------------------------------------------------------------------------------------
def figure_curves():
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6))
    for r, (corpus, title) in enumerate(CORPORA.items()):
        sc = read(f"scores_{corpus}.csv")
        y = sc.nil.values
        for key, (lab, col, ls) in CURVES.items():
            fpr, tpr, _ = roc_curve(y, sc[key])
            prec, rec, _ = precision_recall_curve(y, sc[key])
            lw = 2.0 if key == PRIMARY else 1.3
            ax[r, 0].plot(fpr, tpr, color=col, ls=ls, lw=lw, label=lab)
            ax[r, 1].plot(rec, prec, color=col, ls=ls, lw=lw, label=lab)
        ax[r, 0].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
        ax[r, 1].axhline(y.mean(), color=INK2, lw=0.8, ls=":")
        ax[r, 1].text(0.35, y.mean() - 0.012, f"chance level (prevalence {y.mean():.1%})",
                      ha="left", va="top", color=INK2, fontsize=7)
        ax[r, 0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"{title}: ROC")
        ax[r, 1].set(xlabel="Recall", ylabel="Precision", title=f"{title}: precision-recall", ylim=(0, 1))
        for a in ax[r]:
            a.title.set_fontsize(8.5)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "fig_roc_pr.pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_effort():
    """Share of mentions without a concept found against the share of mentions read."""
    tri = read("results_triage.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for i, (corpus, title) in enumerate(CORPORA.items()):
        sc = read(f"scores_{corpus}.csv")
        y = sc.nil.values
        for key, (lab, col, ls) in CURVES.items():
            if key == "length":
                continue
            order = np.argsort(-sc[key].values, kind="stable")
            found = np.concatenate([[0], np.cumsum(y[order]) / y.sum()])
            read_ = np.arange(len(y) + 1) / len(y)
            ax[i].plot(read_, found, color=col, ls=ls, lw=2.0 if key == PRIMARY else 1.3, label=lab)
        ax[i].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":", label="Random order")
        d = tri[tri.corpus == corpus]
        w = d[d.run.str.startswith("Ward,")]
        h = d[d.run.str.startswith("HDBSCAN, seed")]
        ax[i].scatter(w.covered, w.captured, marker="D", s=22, color=INK, zorder=5,
                      label="Ghost Clusters, Ward (10-50 per cluster)")
        ax[i].scatter(h.covered, h.captured, marker="o", s=18, facecolor="white", edgecolor=INK, zorder=5,
                      label="Ghost Clusters, HDBSCAN (5 seeds)")
        ax[i].set(xlabel="Share of mentions read", ylabel="Share of no-concept mentions found", title=title,
                  xlim=(0, 1), ylim=(0, 1.01))
        ax[i].title.set_fontsize(8.5)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "fig_review_effort.pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_forest():
    res = read("results_human_labels.csv")
    order = list(res.method.unique())
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.2 + 0.27 * len(order)), sharey=True)
    yy = np.arange(len(order))[::-1]
    for i, (corpus, title) in enumerate(CORPORA.items()):
        d = res[res.corpus == corpus].set_index("method").loc[order]
        ax[i].hlines(yy, d.auprc_lo, d.auprc_hi, color=INK2, lw=1.2)
        ax[i].plot(d.auprc, yy, "o", ms=5, color="#2a78d6", mec="white", mew=1)
        ax[i].axvline(d.prevalence.iloc[0], color=INK2, ls=":", lw=0.8)
        ax[i].axvline(d.loc["string", "auprc"], color="#eb6834", ls=(0, (4, 2)), lw=1)
        ax[i].set(title=title, xlabel="AUPRC (95% bootstrap CI)")
        ax[i].title.set_fontsize(8.5)
    ax[0].set_yticks(yy)
    ax[0].set_yticklabels([label(m) for m in order], fontsize=7)
    fig.text(0.99, 0.01, "dotted: prevalence (chance);  dashed: string-match baseline", ha="right", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "fig_auprc_forest.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def table_coverage():
    c = read("coverage.csv")
    body = [f"{r.corpus} & {r.distinct:,} & {pct(r.pt)} & {pct(r.llt)} & {pct(r.pt_or_llt)} & {pct(r.stemmed)} \\\\"
            for r in c.itertuples()]
    write("tab_coverage.tex", table(
        "Share of distinct patient expressions that are verbatim MedDRA 24.0 term names after lower-casing and "
        "punctuation removal (strict criterion), and after additionally ignoring inflection, word order and function "
        "words (relaxed criterion). Distinct expressions are counted after this normalisation.",
        "tab:coverage", "@{}lrrrrr@{}",
        r" & & \multicolumn{3}{c}{Strict} & Relaxed \\" "\n" r"\cmidrule(lr){3-5}\cmidrule(l){6-6}" "\n"
        "Corpus & Distinct expressions & PT name & LLT name & PT or LLT & PT or LLT", body))


def table_human():
    res = read("results_human_labels.csv").set_index(["corpus", "method"])
    base = read("results_baselines.csv").set_index(["corpus", "method"])
    rows = [("res", "length", "Length (words)"), ("res", "string", "String match (token-sort ratio)"),
            ("base", "TF-IDF, character 3-5-grams", "TF-IDF, character 3--5-grams"),
            ("base", "BM25", "BM25"),
            ("res", "sapbert|max|PT+LLT", "Dense normaliser confidence (SapBERT, top-1)"), "mid"]
    rows += [("res", f"{e}|top5|PT+LLT", f"{ENCODER_NAMES[e]}, top-5") for e in ENCODERS]
    rows += ["mid", ("res", PRIMARY, "Consensus gap (label-free, primary)"), "mid",
             ("res", "hybrid", "Hybrid (supervised, secondary)")]
    lines = []
    for item in rows:
        if item == "mid":
            lines.append(r"\midrule")
            continue
        src, key, name = item
        t = res if src == "res" else base
        cells = " & ".join(f"{t.loc[(c, key)].auroc:.3f} & {ci(t.loc[(c, key)].to_frame().T.itertuples().__next__())}"
                           for c in CORPORA)
        line = f"{name} & {cells} \\\\"
        lines.append(bold_row(line[:-3]) + r" \\" if key == PRIMARY else line)
    prev = read("results_human_labels.csv").groupby("corpus").prevalence.first()
    write("tab_human.tex", table(
        r"Detection of mentions that annotators left without a terminology concept. CADEC: MedDRA CONCEPT\_LESS "
        f"(prevalence {pct(prev['cadec'])}); PsyTAR: no SNOMED CT code ({pct(prev['psytar'])}). Encoder rows use the "
        r"top-5 mean similarity against MedDRA PTs and LLTs; the consensus averages their percentile ranks and has "
        r"no fitted parameters. The hybrid is fitted on the corpus labels (out-of-fold predictions). 95\% bootstrap "
        r"confidence intervals over documents. All variants are listed in Supplementary Table~S1.",
        "tab:human", "@{}lcccc@{}",
        r" & \multicolumn{2}{c}{CADEC} & \multicolumn{2}{c}{PsyTAR} \\" "\n" r"\cmidrule(lr){2-3}\cmidrule(l){4-5}" "\n"
        r"Method & AUROC & AUPRC [95\% CI] & AUROC & AUPRC [95\% CI]", lines, extra="\\setlength{\\tabcolsep}{3.5pt}\n"))


def table_controls():
    L = read("results_length.csv")
    V = read("results_score_variants.csv")
    C = read("results_label_checks.csv")
    T = read("results_human_labels.csv").set_index(["corpus", "method"])

    def cells(get):
        return " & ".join(f"{a:.3f} & {b:.3f}" for a, b in (get(c) for c in CORPORA))

    def pair(df, c, **kw):
        r = df[(df.corpus == c) & np.logical_and.reduce([df[k] == v for k, v in kw.items()])].iloc[0]
        return r.auroc, r.auprc

    lines = []
    for key, name in [("string", "String match"), (PRIMARY, "Consensus gap")]:
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{name}}}}} \\")
        lines.append(f"\\quad All mentions & {cells(lambda c: (T.loc[(c, key)].auroc, T.loc[(c, key)].auprc))} \\\\")
        for s in ["1-2 words", "3-4 words", "5+ words"]:
            lines.append(f"\\quad {s.replace('-', '--')} & {cells(lambda c: pair(L, c, stratum=s, method=key))} \\\\")
        lines.append(f"\\quad Effect of length removed & {cells(lambda c: pair(L, c, stratum='all, length removed', method=key))} \\\\")
        lines.append(f"\\quad Distinct expressions & {cells(lambda c: pair(C, c, level='distinct expression', method=key))} \\\\")
        lines.append(f"\\quad Verbatim-term positives removed & "
                     f"{cells(lambda c: pair(C, c, subset='verbatim-term positives removed', method=key))} \\\\")
        if key == PRIMARY:
            for v, n in [("top-1", "Top-1 instead of top-5"), ("top-10", "Top-10 instead of top-5"),
                         ("concept-level top-5", "One similarity per parent PT")]:
                lines.append(f"\\quad {n} & {cells(lambda c: pair(V, c, variant=v, encoder=PRIMARY))} \\\\")
            for t in ["low", "middle", "high"]:
                lines.append(f"\\quad Nearest PT with {t} LLT count (tertile) & "
                             f"{cells(lambda c: pair(V, c, variant=f'LLT density {t}', encoder=PRIMARY))} \\\\")
    sup = L[L.stratum == "all, supervised"]
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{@{}l}{\emph{Supervised models with and without the semantic score}} \\")
    for m in ["length only", "length + string", "length + consensus"]:
        lines.append(f"\\quad {m.capitalize()} & {cells(lambda c: pair(sup, c, method=m))} \\\\")
    write("tab_controls.tex", table(
        r"Robustness of the consensus gap. Length strata are in words; ``effect of length removed'' subtracts the "
        r"least-squares fit of the score on log length (no labels involved). At the distinct-expression level an "
        r"expression is positive when most of its mentions are. ``Verbatim-term positives removed'' drops positive "
        r"mentions that are verbatim MedDRA term names. Supervised models: class-balanced logistic regression with "
        r"five folds grouped by document.",
        "tab:controls", "@{}lcccc@{}",
        r" & \multicolumn{2}{c}{CADEC} & \multicolumn{2}{c}{PsyTAR} \\" "\n" r"\cmidrule(lr){2-3}\cmidrule(l){4-5}" "\n"
        "Analysis & AUROC & AUPRC & AUROC & AUPRC", lines, extra="\\setlength{\\tabcolsep}{4pt}\n"))


def table_supervised():
    s = read("results_supervised.csv")
    order = ["Supervised TF-IDF classifier", "Supervised embedding classifier (BioLORD-2023)",
             "Hybrid (supervised)", "Consensus gap (label-free)"]
    lines = []
    for m in order:
        d = s[s.method == m]
        get = lambda tr, te, st: d[(d.train == tr) & (d.test == te) & (d.setting.str.startswith(st))].iloc[0]
        cells = []
        for tr, te, st in [("cadec", "cadec", "within corpus, folds by documents"),
                           ("cadec", "cadec", "within corpus, folds by expressions"),
                           ("psytar", "psytar", "within corpus, folds by documents"),
                           ("psytar", "psytar", "within corpus, folds by expressions"),
                           ("psytar", "cadec", "transfer"), ("cadec", "psytar", "transfer")]:
            r = get(tr, te, st)
            cells.append(f"{r.auprc:.3f} ({r.auroc:.2f})")
        name = m.replace("(BioLORD-2023)", "(BioLORD)")
        line = f"{name} & " + " & ".join(cells)
        lines.append((bold_row(line) if "label-free" in m else line) + r" \\")
    prev = read("results_human_labels.csv").groupby("corpus").prevalence.first()
    write("tab_supervised.tex", table(
        r"Supervised detectors against the label-free consensus gap: AUPRC (AUROC). Within a corpus, five-fold "
        r"cross-validation with folds grouped by document or by normalised expression, so that repeated expressions "
        r"cannot appear in both training and test data. Transfer: trained on all labels of one corpus and applied "
        r"to the other without refitting. The consensus gap uses no labels, so its value is the same in every "
        f"setting. Chance level: CADEC {prev['cadec']:.3f}, PsyTAR {prev['psytar']:.3f}.",
        "tab:supervised", "@{}lcccccc@{}",
        r" & \multicolumn{2}{c}{Within CADEC} & \multicolumn{2}{c}{Within PsyTAR} & \multicolumn{2}{c}{Across corpora} \\"
        "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}" "\n"
        r"Detector & by document & by expression & by document & by expression & PsyTAR$\rightarrow$CADEC & CADEC$\rightarrow$PsyTAR",
        lines, star=True, size=r"\footnotesize", extra="\\setlength{\\tabcolsep}{4pt}\n"))


def table_operating():
    o = read("results_operating_points.csv")
    order = ["String match (token-sort ratio)", "Dense normaliser confidence (SapBERT, top-1)",
             "Hybrid (supervised)", "Consensus gap (label-free)"]
    lines = []
    for m in order:
        cells = []
        for c in CORPORA:
            r = o[(o.corpus == c) & (o.method == m)].iloc[0]
            cells += [f"{r.p_at_100:.2f}", f"{r.precision_at_recall_50:.2f}", pct(r.read_for_recall_50, 0), pct(r.read_for_recall_80, 0)]
        line = f"{m.replace(' (token-sort ratio)', '').replace(' (SapBERT, top-1)', '')} & " + " & ".join(cells)
        lines.append((bold_row(line) if "label-free" in m else line) + r" \\")
    write("tab_operating.tex", table(
        r"Operating points for review. P@100: precision among the 100 highest-scored mentions. P\,@\,R50: precision "
        r"at 50\% recall. Read for R50/R80: share of all mentions a reviewer must read, in score order, to find 50\% "
        r"or 80\% of the mentions without a concept. Supervised classifiers are compared in Supplementary Table~S6.",
        "tab:operating", "@{}lcccccccc@{}",
        r" & \multicolumn{4}{c}{CADEC} & \multicolumn{4}{c}{PsyTAR} \\" "\n" r"\cmidrule(lr){2-5}\cmidrule(l){6-9}" "\n"
        r"Score & P@100 & P\,@\,R50 & Read for R50 & Read for R80 & P@100 & P\,@\,R50 & Read for R50 & Read for R80",
        lines, star=True, extra="\\setlength{\\tabcolsep}{4pt}\n"))


def table_hidden():
    h = read("results_hidden_concepts.csv")
    h = h[h.encoder == PRIMARY]
    names = {"PT": "Single PTs (siblings kept)", "HLT": "Whole HLTs", "HLGT": "Whole HLGTs"}
    lines = []
    for corpus in ["smm4h", "cadec"]:
        for reg in ["PT", "HLT", "HLGT"]:
            d = h[(h.corpus == corpus) & (h.regime == reg)].sort_values("fraction")
            first = NAME[corpus] if reg == "PT" else ""
            lines.append(f"{first} & {names[reg]} & " + " & ".join(
                f"${r.auroc:.3f} \\pm {r.auroc_sd:.3f}$ & {r.auprc:.3f} ({100 * r.positives:.0f}\\%)" for r in d.itertuples()) + r" \\")
        if corpus == "smm4h":
            lines.append(r"\midrule")
    write("tab_hidden.tex", table(
        r"Hidden-concept simulation. A random share of concepts is removed from MedDRA with all of their LLTs and "
        r"descendant PTs; every distinct (expression, gold concept) pair is scored with the consensus gap against "
        r"the reduced terminology, and pairs whose gold PT was removed are the positives. AUROC (mean $\pm$ s.d.\ "
        r"over ten removals) and AUPRC (share of positive pairs). Per-encoder results are in Supplementary Table~S7.",
        "tab:hidden", "@{}llcccccc@{}",
        r" & & \multicolumn{2}{c}{10\% removed} & \multicolumn{2}{c}{20\% removed} & \multicolumn{2}{c}{30\% removed} \\"
        "\n" r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}" "\n"
        r"Corpus & Removed unit & AUROC & AUPRC & AUROC & AUPRC & AUROC & AUPRC",
        lines, star=True, extra="\\setlength{\\tabcolsep}{4pt}\n"))


def table_triage():
    t = read("results_triage.csv")
    ref = f"Ward, {WARD_MEAN_SIZE} per cluster"
    lines = []

    def rng(s, f=lambda v: f"{v:.0f}"):
        return f"{f(s.min())}--{f(s.max())}" if s.min() != s.max() else f(s.min())

    for corpus in CORPORA:
        d = t[t.corpus == corpus]
        groups = [(f"Ward, {WARD_MEAN_SIZE} per cluster (primary)", d[d.run == ref]),
                  ("Ward, 10--50 per cluster", d[d.run.str.startswith("Ward,")]),
                  ("Ward, other encoders", d[d.run.str.startswith("Ward on")]),
                  ("HDBSCAN, 5 seeds", d[d.run.str.startswith("HDBSCAN, seed")])]
        for i, (name, g) in enumerate(groups):
            p = lambda s: rng(s * 100, lambda v: f"{v:.1f}") + r"\%"
            cells = [rng(g.ghost_clusters), rng(g.expressions), p(g.covered),
                     rng(g.enrichment, lambda v: f"{v:.1f}") + r"$\times$", p(g.captured),
                     p(g.captured_by_ranking), rng(g.ranked_review_units)]
            line = f"{NAME[corpus] if i == 0 else ''} & {name} & " + " & ".join(cells)
            lines.append((bold_row(line) if i == 0 else line) + r" \\")
        if corpus == "cadec":
            lines.append(r"\midrule")
    write("tab_triage.tex", table(
        r"Ghost Clusters as review units. Ghost Clusters are the quartile of clusters (at least three expressions) "
        r"with the highest mean consensus gap. Enrichment: rate of no-concept mentions inside Ghost Clusters divided "
        r"by the rate outside. Ranked list: share found by reading the same number of top-ranked mentions instead, "
        r"and the number of clusters those mentions fall into (``rank, then group''). Ranges are over the runs in "
        r"each row; per-run results are in Supplementary Table~S8.",
        "tab:triage", "@{}llrrrrrrr@{}",
        r" & & & & & & & \multicolumn{2}{c}{Ranked list} \\" "\n" r"\cmidrule(l){8-9}" "\n"
        r"Corpus & Clustering & Ghost Clusters & Expressions & Mentions covered & Enrichment & Found & Found & Review units",
        lines, star=True, size=r"\footnotesize", extra="\\setlength{\\tabcolsep}{3.5pt}\n"))


def latex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s


def shorten(s, n=34):
    """Shorten to at most n characters at a word boundary and escape for LaTeX."""
    s = " ".join(str(s).split())
    if len(s) <= n:
        return latex_escape(s)
    return latex_escape(s[:s[:n].rfind(" ")]) + r" \dots"


def table_ghosts():
    notes = pd.read_csv(os.path.join(ROOT, "annotations", "ghost_cluster_assessment.csv"))
    lines = []
    for corpus in CORPORA:
        g = read(f"ghost_clusters_{corpus}_ward.csv").set_index("cluster")
        for r in notes[notes.corpus == corpus].itertuples():
            row = g.loc[r.cluster]
            assert row.examples.split(" | ")[0] == r.first_example, f"cluster {corpus}/{r.cluster} changed"
            ex = "; ".join(shorten(x) for x in row.examples.split(" | ")[:2])
            lines.append(f"{NAME[corpus]} & {ex} & {row.expressions} & {100 * row.share_unmappable:.0f}\\% & "
                         f"{latex_escape(r.closest_adequate_meddra_term)} & {r.gap_type} \\\\")
        if corpus == "cadec":
            lines.append(r"\midrule")
    write("tab_ghosts.tex", table(
        r"The eight Ward Ghost Clusters most distant from MedDRA in each labelled corpus, with the number of "
        r"distinct expressions, the share of member mentions left without a concept by the annotators, the closest "
        r"adequate MedDRA~24.0 term found on inspection of the terms nearest to the cluster centroid, and the "
        r"authors' classification of the gap (Section~\ref{ssec:content}).",
        "tab:ghosts", r"@{}lL{5.3cm}rrL{4.3cm}L{2.3cm}@{}",
        r"Corpus & Most frequent expressions & Expr. & No concept & Closest adequate MedDRA term & Gap type",
        lines, star=True, extra="\\setlength{\\tabcolsep}{3.5pt}\n"))


# ---------------------------------------------------------------------------------------------
def supplementary():
    parts = []
    res = read("results_human_labels.csv").merge(read("results_gain_over_string.csv"), on=["corpus", "method"], how="left")
    lines = []
    for r in res.itertuples():
        gain = "--" if pd.isna(r.gain) else f"${r.gain:+.3f}$ [${r.lo:+.3f}$, ${r.hi:+.3f}$] & {r.p_holm:.3f}"
        if gain == "--":
            gain = "-- & --"
        lines.append(f"{NAME[r.corpus]} & {label(r.method)} & {ci(r, 'auroc')} & {ci(r)} & {gain} \\\\")
    parts.append(table(
        r"Supplementary Table S1. All scoring variants. AUPRC gain: paired bootstrap difference from the "
        r"string-matching baseline (mean and 95\% interval) with Holm-adjusted bootstrap $p$-values over all 28 "
        r"comparisons.", "tab:s1", "@{}llcccc@{}",
        r"Corpus & Method & AUROC [95\% CI] & AUPRC [95\% CI] & AUPRC gain over string & $p_{\mathrm{Holm}}$", lines,
        size=r"\scriptsize"))

    pc = read("results_primary_comparisons.csv")
    lines = [f"{NAME[r.corpus]} & {label(r.versus)} & ${r.gain:+.3f}$ [${r.lo:+.3f}$, ${r.hi:+.3f}$] & {r.p:.3f} \\\\" for r in pc.itertuples()]
    parts.append(table(r"Supplementary Table S2. Paired AUPRC differences between the consensus gap and other "
                       r"scores (consensus minus comparator).", "tab:s2", "@{}llcc@{}",
                       r"Corpus & Comparator & AUPRC difference [95\% CI] & $p$", lines))

    b = read("results_baselines.csv")
    lines = [f"{NAME[r.corpus]} & {r.method} & {ci(r, 'auroc')} & {ci(r)} \\\\" for r in b.itertuples()]
    parts.append(table(r"Supplementary Table S3. Additional baselines. Supervised classifiers: five-fold "
                       r"cross-validation grouped by document (class-balanced logistic regression on character "
                       r"2--5-gram TF-IDF features or on frozen sentence embeddings).", "tab:s3", "@{}llcc@{}",
                       r"Corpus & Method & AUROC [95\% CI] & AUPRC [95\% CI]", lines, size=r"\scriptsize"))

    L = read("results_length.csv")
    lines = [f"{NAME[r.corpus]} & {r.stratum.replace('-', '--')} & {r.method} & {r.n} & {pct(r.prevalence)} & {r.auroc:.3f} & {r.auprc:.3f} \\\\"
             for r in L.itertuples()]
    parts.append(table(r"Supplementary Table S4. Length analyses.", "tab:s4", "@{}lllrrrr@{}",
                       "Corpus & Stratum & Score & Mentions & Prevalence & AUROC & AUPRC", lines, size=r"\scriptsize"))

    V = read("results_score_variants.csv")
    lines = [f"{NAME[r.corpus]} & {r.variant} & {label(r.encoder) if r.encoder == PRIMARY else ENCODER_NAMES[r.encoder]} & {r.auroc:.3f} & {r.auprc:.3f} \\\\"
             for r in V.itertuples()]
    parts.append(table(r"Supplementary Table S5. Definition of the semantic gap: number of nearest terms averaged, "
                       r"concept-level aggregation (at most one similarity per parent PT) and terminology density "
                       r"(number of LLTs under the nearest PT, BioLORD-2023, tertiles).", "tab:s5", "@{}lllrr@{}",
                       "Corpus & Variant & Encoder & AUROC & AUPRC", lines, size=r"\scriptsize"))

    o = read("results_operating_points.csv")
    lines = [f"{NAME[r.corpus]} & {r.method} & {r.p_at_50:.2f} & {r.p_at_100:.2f} & {r.p_at_200:.2f} & {r.p_at_500:.2f} & "
             f"{r.precision_at_recall_50:.2f} & {r.precision_at_recall_80:.2f} & {pct(r.read_for_recall_50, 0)} & {pct(r.read_for_recall_80, 0)} \\\\"
             for r in o.itertuples()]
    parts.append(table(r"Supplementary Table S6. Operating points for every detector (supervised classifiers: "
                       r"out-of-fold predictions within the corpus).", "tab:s6", "@{}llrrrrrrrr@{}",
                       r"Corpus & Detector & P@50 & P@100 & P@200 & P@500 & P\,@\,R50 & P\,@\,R80 & Read R50 & Read R80",
                       lines, size=r"\scriptsize"))

    h = read("results_hidden_concepts.csv")
    lines = [f"{NAME[r.corpus]} & {r.regime} & {int(100 * r.fraction)}\\% & {label(r.encoder) if r.encoder == PRIMARY else ENCODER_NAMES[r.encoder]} & "
             f"${r.auroc:.3f} \\pm {r.auroc_sd:.3f}$ & ${r.auprc:.3f} \\pm {r.auprc_sd:.3f}$ \\\\" for r in h.itertuples()]
    parts.append(table(r"Supplementary Table S7. Hidden-concept simulation for every encoder (top-5 gap).",
                       "tab:s7", "@{}lllrrr@{}", "Corpus & Removed unit & Share & Encoder & AUROC & AUPRC", lines,
                       size=r"\scriptsize"))

    t = read("results_triage.csv")
    lines = [f"{NAME[r.corpus]} & {r.run.replace('sapbert', 'SapBERT').replace('biolord', 'BioLORD-2023')} & {r.clusters} & "
             f"{r.ghost_clusters} & {r.expressions} & {pct(r.covered)} & ${r.enrichment:.2f}\\times$ & {pct(r.captured)} & "
             f"{pct(r.ghosts_majority_unmappable, 0)} & {pct(r.group_decision_agreement, 0)} & {pct(r.captured_by_ranking)} & {r.ranked_review_units} \\\\"
             for r in t.itertuples()]
    st = read("results_stability.csv")
    stab = [f"{NAME[r.corpus]} & {r.method} & {r.ari:.3f} & {r.jaccard:.3f} ({r.jaccard_min:.3f}) \\\\" for r in st.itertuples()]
    cs = read("results_cluster_scoring.csv")
    blend = [f"{NAME[r.corpus]} & {'Ward' if r.clustering == 'ward' else 'HDBSCAN'} & {label(r.score)} & {r.auprc_expression:.3f} & "
             f"{r.auprc_blend:.3f} & ${r.gain:+.3f}$ [${r.lo:+.3f}$, ${r.hi:+.3f}$] \\\\" for r in cs.itertuples()]
    parts.append(table(r"Supplementary Table S8. Ghost Cluster triage for every clustering run. Majority: share of "
                       r"Ghost Clusters in which most mentions lack a concept. Agreement: share of mentions in Ghost "
                       r"Clusters whose label equals their cluster's majority label (accuracy of a decision taken "
                       r"per cluster). Ranked: share found, and review units formed, by reading the same number of "
                       r"top-ranked mentions.", "tab:s8", "@{}llrrrrrrrrrr@{}",
                       r"Corpus & Run & Clusters & Ghost & Expr. & Covered & Enrich. & Found & Majority & Agreement & Ranked & Units",
                       lines, size=r"\scriptsize"))
    parts.append(table(r"Supplementary Table S9. Stability (top): mean pairwise adjusted Rand index of partitions "
                       r"and mean (minimum) Jaccard overlap of flagged expressions. Cluster-informed scoring "
                       r"(bottom): each mention's score blended equally with its cluster's mean score.", "tab:s9",
                       "@{}llrr@{}", r"Corpus & Comparison & ARI & Flagged-expression Jaccard", stab)
                 .replace("\\end{tabular}\n\\end{table}", "\\end{tabular}\n\n\\vspace{1em}\n\\begin{tabular}{@{}lllrrc@{}}\n\\toprule\n"
                          "Corpus & Clustering & Score & AUPRC & AUPRC blended & Difference [95\\% CI] \\\\\n\\midrule\n"
                          + "\n".join(blend) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"))

    g = read("smm4h_grid.csv")
    lines = [f"{r.min_cluster_size} & {r.min_samples} & {r.selection} & {r.clusters} & {pct(r.noise)} & {r.silhouette:.3f} & {r.dbcv:.3f} \\\\"
             for r in g.itertuples()]
    parts.append(table(r"Supplementary Table S10. HDBSCAN configuration grid on SMM4H-2017 (S-PubMedBERT "
                       r"embeddings, UMAP to five dimensions). The configuration used (minimum cluster size 7, minimum "
                       r"samples 7, excess of mass) has the highest DBCV.", "tab:s10", "@{}rrlrrrr@{}",
                       r"Min.\ cluster size & Min.\ samples & Selection & Clusters & Noise & Silhouette & DBCV", lines))

    with open(os.path.join(OUT_DIR, "descriptives.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    flow = [
        f"CADEC & Posts in the MedDRA annotation layer & {d['cadec_posts_total']:,} \\\\",
        f" & Spans of entity type ADR (used) & {d['cadec_mentions']:,} from {d['cadec_posts']:,} posts \\\\",
        f" & CONCEPT\\_LESS spans & {d['cadec_nil']} \\\\",
        f" & CONCEPT\\_LESS spans that are verbatim MedDRA names & {d['cadec_conceptless_verbatim_meddra']} "
        f"({d['cadec_conceptless_verbatim_llt_only']} LLT only) \\\\",
        r"\midrule",
        f"PsyTAR & Mapped ADR mentions (used) & {d['psytar_mentions']:,} from {d['psytar_reviews']} reviews \\\\",
        f" & Without SNOMED CT code & {d['psytar_nil']} \\\\",
        f" & \\quad of which the chosen UMLS concept has no SNOMED CT code & {d['psytar_uncoded_with_umls_concept']} \\\\",
        f" & \\quad of which a related SNOMED CT concept is given as fallback & {d['psytar_uncoded_with_fallback_snomed']} \\\\",
        f" & \\quad of which neither a UMLS concept nor a fallback is given & {d['psytar_uncoded_without_any_concept']} \\\\",
        f" & Of these, verbatim MedDRA names & {d['psytar_uncoded_verbatim_meddra']} ({d['psytar_uncoded_verbatim_llt_only']} LLT only) \\\\",
        r"\midrule",
        f"SMM4H-2017 & Mentions after removal of one header line & {d['smm4h_mentions']:,} \\\\",
        f" & Distinct raw strings / after normalisation & {d['smm4h_distinct_strings']:,} / {d['smm4h_distinct_normalised']:,} \\\\",
        f" & Gold code at PT / LLT / other level & {d['smm4h_gold_levels'].get('PT', 0):,} / {d['smm4h_gold_levels'].get('LLT', 0)} / "
        f"{d['smm4h_gold_levels'].get('none', 0)} \\\\",
    ]
    parts.append(table(r"Supplementary Table S11. Data flow. Distinct raw strings are the units that are embedded; "
                       r"Table~2 of the main text counts distinct expressions after lower-casing and punctuation "
                       r"removal. Posts of CADEC without any ADR span do not contribute mentions.", "tab:s11",
                       r"@{}llr@{}", "Corpus & Step & Count", flow))
    lc = read("results_label_checks.csv")
    lc = lc[lc.level == "mention"]
    wide = lc.pivot_table(index=["corpus", "method"], columns="subset", values=["auroc", "auprc"])
    lines = []
    for (corpus, method), r in wide.iterrows():
        lines.append(f"{NAME[corpus]} & {label(method)} & {r[('auroc', 'all mentions')]:.3f} & {r[('auprc', 'all mentions')]:.3f} & "
                     f"{r[('auroc', 'verbatim-term positives removed')]:.3f} & {r[('auprc', 'verbatim-term positives removed')]:.3f} \\\\")
    parts.append(table(r"Supplementary Table S12. Effect of removing positive mentions that are verbatim MedDRA~24.0 "
                       r"term names (CADEC: 10 mentions; PsyTAR: 71 mentions, 70 of them LLTs).", "tab:s12",
                       "@{}llrrrr@{}",
                       r" & & \multicolumn{2}{c}{All mentions} & \multicolumn{2}{c}{Verbatim positives removed} \\"
                       "\n" r"\cmidrule(lr){3-4}\cmidrule(l){5-6}" "\n"
                       "Corpus & Score & AUROC & AUPRC & AUROC & AUPRC", lines, size=r"\scriptsize"))
    write("supp_tables.tex", "\n".join(p.replace("[htbp!]", "[p]") for p in parts))


def main():
    figure_curves()
    figure_effort()
    figure_forest()
    table_coverage()
    table_human()
    table_controls()
    table_supervised()
    table_operating()
    table_hidden()
    table_triage()
    table_ghosts()
    supplementary()
    print("figures:", sorted(os.listdir(FIG_DIR)))
    print("tables:", sorted(os.listdir(TAB_DIR)))


if __name__ == "__main__":
    main()
