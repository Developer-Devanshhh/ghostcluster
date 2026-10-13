"""Step 5 - figures (PDF) and LaTeX tables of the manuscript and its Supplementary Information.

Outputs: outputs/figures/*.pdf and outputs/tables/*.tex (included verbatim by the manuscript).
Colours are a colour-vision-deficiency-checked categorical palette; line styles differ as well,
so methods are never distinguished by colour alone.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ghostcluster.config import ENCODER_NAMES, ENCODERS, FIG_DIR, OUT_DIR, TAB_DIR

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42})
CORPORA = {"cadec": "CADEC (MedDRA, CONCEPT_LESS)", "psytar": "PsyTAR (no SNOMED CT code)"}
CURVES = {
    "length": ("Length (words)", "#2a78d6", (0, (1, 1.5))),
    "string": ("String match (PT+LLT)", "#eb6834", (0, (4, 2))),
    "spubmedbert|max|PT": ("S-PubMedBERT, max, PT only", "#1baf7a", (0, (6, 1.5, 1, 1.5))),
    "spubmedbert|top5|PT+LLT": ("S-PubMedBERT, top-5, PT+LLT", "#eda100", (0, (8, 2))),
    "hybrid": ("Hybrid (primary)", "#e87ba4", "solid"),
}


def label(key):
    if key == "length":
        return "Length (words)"
    if key == "string":
        return "String match (PT+LLT)"
    if key == "hybrid":
        return "Hybrid (primary)"
    enc, kind, ref = key.split("|")
    return f"{ENCODER_NAMES[enc]}, {'top-5 mean' if kind == 'top5' else 'max'}, {'PT only' if ref == 'PT' else 'PT+LLT'}"


def read(name):
    return pd.read_csv(os.path.join(OUT_DIR, name))


def write(name, text):
    with open(os.path.join(TAB_DIR, name), "w", encoding="utf-8") as fh:
        fh.write(text)


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


# ---------------------------------------------------------------------------------------------
def figure_curves():
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6))
    for r, (corpus, title) in enumerate(CORPORA.items()):
        sc = read(f"scores_{corpus}.csv")
        y = sc.nil.values
        for key, (lab, col, ls) in CURVES.items():
            fpr, tpr, _ = roc_curve(y, sc[key])
            prec, rec, _ = precision_recall_curve(y, sc[key])
            ax[r, 0].plot(fpr, tpr, color=col, ls=ls, lw=1.5, label=lab)
            ax[r, 1].plot(rec, prec, color=col, ls=ls, lw=1.5, label=lab)
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
    body = "\n".join(f"{r.corpus} & {r.distinct:,} & {pct(r.pt)} & {pct(r.llt)} & {pct(r.pt_or_llt)} & {pct(r.stemmed)} \\\\"
                     for r in c.itertuples())
    write("tab_coverage.tex", r"""\begin{table}[htbp!]
\caption{Share of distinct patient expressions that are verbatim MedDRA 24.0 term names after lower-casing and punctuation removal (strict criterion), and after additionally ignoring inflection, word order and function words (relaxed criterion).}
\label{tab:coverage}
\centering\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{Strict} & Relaxed \\
\cmidrule(lr){3-5}\cmidrule(l){6-6}
Corpus & Distinct expressions & PT name & LLT name & PT or LLT & PT or LLT \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_human():
    res = read("results_human_labels.csv").set_index(["corpus", "method"])
    rows = [("length", None), ("string", None), ("spubmedbert|max|PT", None), "mid"]
    rows += [(f"{e}|top5|PT+LLT", ENCODER_NAMES[e]) for e in ENCODERS] + ["mid", ("hybrid", None)]
    lines = []
    for item in rows:
        if item == "mid":
            lines.append(r"\midrule")
            continue
        key, name = item
        cells = []
        for corpus in CORPORA:
            r = res.loc[(corpus, key)]
            cells.append(f"{r.auroc:.3f} & {r.auprc:.3f} [{r.auprc_lo:.3f}, {r.auprc_hi:.3f}]")
        text = f"{name or label(key)} & " + " & ".join(cells) + r" \\"
        if key == "hybrid":
            text = r"\textbf{Hybrid (primary)} & " + " & ".join(r"\textbf{" + c.replace(" & ", r"} & \textbf{") + "}" for c in cells) + r" \\"
        lines.append(text)
    prev = read("results_human_labels.csv").groupby("corpus").prevalence.first()
    write("tab_human.tex", r"""\begin{table}[htbp!]
\caption{Detection of mentions that annotators judged to have no terminology concept. CADEC: MedDRA CONCEPT\_LESS (prevalence """ + pct(prev["cadec"]) + r"""); PsyTAR: no SNOMED CT code (""" + pct(prev["psytar"]) + r"""). Encoder rows use the top-5 mean similarity against MedDRA PTs and LLTs. 95\% bootstrap confidence intervals over documents. Results for all scoring variants are given in Supplementary Table~S1.}
\label{tab:human}
\centering\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{CADEC} & \multicolumn{2}{c}{PsyTAR} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Method & AUROC & AUPRC [95\% CI] & AUROC & AUPRC [95\% CI] \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_hidden():
    h = read("results_hidden_concepts.csv")
    lines = []
    for e in ENCODERS:
        d = h[h.encoder == e].sort_values("hidden")
        lines.append(f"{ENCODER_NAMES[e]} & " + " & ".join(f"${r.auroc:.3f} \\pm {r.auroc_sd:.3f}$" for r in d.itertuples()) + r" \\")
    write("tab_hidden.tex", r"""\begin{table}[htbp!]
\caption{Hidden-concept simulation on SMM4H-2017: AUROC (mean $\pm$ s.d.\ over ten repeats) for separating expressions whose gold PT (and its LLTs) was removed from the terminology from those whose concept remains. Maximum similarity against the remaining PTs and LLTs.}
\label{tab:hidden}
\centering\small
\begin{tabular}{@{}lccc@{}}
\toprule
Encoder & 10\% hidden & 20\% hidden & 30\% hidden \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_triage():
    t = read("results_triage.csv")
    lines = []
    for corpus, name in [("cadec", "CADEC"), ("psytar", "PsyTAR")]:
        d = t[t.corpus == corpus]
        hd = d[d.run.str.startswith("HDBSCAN, seed")]
        w = d[d.run.str.startswith("Ward")].reset_index(drop=True)
        k = lambda r: r.run.split("= ")[1]
        lines.append(rf"\multirow{{3}}{{*}}{{{name}}} & HDBSCAN (5 seeds) & {hd.ghost_clusters.min()}--{hd.ghost_clusters.max()} & "
                     rf"{pct(hd.covered.mean())} & ${hd.enrichment.mean():.1f}\times$ ({hd.enrichment.min():.1f}--{hd.enrichment.max():.1f}) & "
                     rf"{pct(hd.captured.mean())} ({100 * hd.captured.min():.1f}--{100 * hd.captured.max():.1f}) & {pct(hd.captured_by_ranking.mean())} \\")
        r0 = w.iloc[0]
        lines.append(rf" & Ward, $K={k(r0)}$ & {r0.ghost_clusters} & {pct(r0.covered)} & ${r0.enrichment:.1f}\times$ & {pct(r0.captured)} & {pct(r0.captured_by_ranking)} \\")
        a, b = w.iloc[1], w.iloc[2]
        lines.append(rf" & Ward, $K={k(a)}$ / ${k(b)}$ & {a.ghost_clusters} / {b.ghost_clusters} & {pct(a.covered)} / {pct(b.covered)} & "
                     rf"${a.enrichment:.1f}\times$ / ${b.enrichment:.1f}\times$ & {pct(a.captured)} / {pct(b.captured)} & "
                     rf"{pct(a.captured_by_ranking)} / {pct(b.captured_by_ranking)} \\")
        if corpus == "cadec":
            lines.append(r"\midrule")
    write("tab_triage.tex", r"""\begin{table*}[t]
\caption{Ghost Clusters as a triage unit. Ghost Clusters are the quartile of clusters (at least three members) most distant from MedDRA. Enrichment: rate of human-judged unmappable mentions inside Ghost Clusters divided by the rate outside. ``Ranking'' is the share captured by reading the same number of top-ranked individual expressions. HDBSCAN values are means over five UMAP seeds (range in parentheses). Per-run results are given in Supplementary Table~S2.}
\label{tab:triage}
\centering\small
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Corpus & Clustering & Ghost Clusters & Mentions covered & Enrichment & Captured & Ranking \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def latex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s


def shorten(s, n=46):
    """Shorten to at most n characters at a word boundary and escape for LaTeX."""
    s = " ".join(str(s).split())
    if len(s) <= n:
        return latex_escape(s)
    return latex_escape(s[:s[:n].rfind(" ")]) + r" \dots"


def table_ghosts():
    lines = []
    for corpus, name in [("cadec", "CADEC"), ("psytar", "PsyTAR")]:
        g = read(f"ghost_clusters_{corpus}.csv").head(5)
        for r in g.itertuples():
            ex = "; ".join(shorten(x) for x in r.examples.split(" | "))
            lines.append(f"{name} & {ex} & {r.size} & {100 * r.share_unmappable:.0f}\\% \\\\")
    write("tab_ghosts.tex", r"""\begin{table*}[t]
\caption{The five Ghost Clusters most distant from MedDRA in each labelled corpus (HDBSCAN, S-PubMedBERT), with the number of distinct expressions and the share of member mentions that annotators judged to have no concept.}
\label{tab:ghosts}
\centering\small
\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{10.6cm}rr@{}}
\toprule
Corpus & Most frequent expressions & Expressions & Judged unmappable \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def supplementary():
    res = read("results_human_labels.csv").merge(read("results_gain_over_string.csv"), on=["corpus", "method"], how="left")
    lines = []
    for r in res.itertuples():
        gain = "--" if pd.isna(r.gain) else f"${r.gain:+.3f}$ [${r.lo:+.3f}$, ${r.hi:+.3f}$]"
        lines.append(f"{'CADEC' if r.corpus == 'cadec' else 'PsyTAR'} & {label(r.method)} & {r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}] & "
                     f"{r.auprc:.3f} [{r.auprc_lo:.3f}, {r.auprc_hi:.3f}] & {gain} \\\\")
    s1 = r"""\begin{table}[p]
\caption{Supplementary Table S1. Detection of mentions judged to have no concept: all scoring variants. AUPRC gain: paired bootstrap difference from the string-matching baseline (mean and 95\% interval).}
\centering\small
\begin{tabular}{@{}llccc@{}}
\toprule
Corpus & Method & AUROC [95\% CI] & AUPRC [95\% CI] & AUPRC gain over string \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    t = read("results_triage.csv")
    lines = [f"{'CADEC' if r.corpus == 'cadec' else 'PsyTAR'} & {r.run.replace('sapbert', 'SapBERT').replace('biolord', 'BioLORD-2023')} & {r.clusters} & {r.ghost_clusters} & {pct(r.covered)} & "
             f"${r.enrichment:.2f}\\times$ & {pct(r.captured)} & {pct(r.captured_by_ranking)} \\\\" for r in t.itertuples()]
    st = read("results_stability.csv")
    stab = [f"{r.corpus.upper() if r.corpus != 'psytar' else 'PsyTAR'} & {r.method} & {r.ari:.3f} & {r.jaccard:.3f} ({r.jaccard_min:.3f}) \\\\"
            for r in st.itertuples()]
    s2 = r"""\begin{table}[p]
\caption{Supplementary Table S2. Ghost Cluster triage for every clustering run (top) and stability (bottom): mean pairwise adjusted Rand index of partitions and mean (minimum) Jaccard overlap of the sets of flagged expressions.}
\centering\small
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
Corpus & Clustering run & Clusters & Ghost Clusters & Covered & Enrichment & Captured & Ranking \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}

\vspace{1em}
\begin{tabular}{@{}llrr@{}}
\toprule
Corpus & Comparison & ARI & Flagged-expression Jaccard \\
\midrule
""" + "\n".join(stab) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    g = read("smm4h_grid.csv")
    lines = [f"{r.min_cluster_size} & {r.min_samples} & {r.selection} & {r.clusters} & {pct(r.noise)} & {r.silhouette:.3f} & {r.dbcv:.3f} \\\\"
             for r in g.itertuples()]
    s3 = r"""\begin{table}[p]
\caption{Supplementary Table S3. HDBSCAN configuration grid on SMM4H-2017 (S-PubMedBERT embeddings, UMAP to five dimensions). The configuration used throughout (minimum cluster size 7, minimum samples 7, excess of mass) is the one with the highest DBCV.}
\centering\small
\begin{tabular}{@{}rrlrrrr@{}}
\toprule
Min.\ cluster size & Min.\ samples & Selection & Clusters & Noise & Silhouette & DBCV \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("supp_tables.tex", s1 + "\n" + s2 + "\n" + s3)


def main():
    figure_curves()
    figure_forest()
    table_coverage()
    table_human()
    table_hidden()
    table_triage()
    table_ghosts()
    supplementary()
    print("figures:", sorted(os.listdir(FIG_DIR)))
    print("tables:", sorted(os.listdir(TAB_DIR)))


if __name__ == "__main__":
    main()
