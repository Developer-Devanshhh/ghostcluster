# GhostCluster

Code for the manuscript

> **GhostCluster: Label-Free Triage of Patient-Reported Adverse Drug Event Expressions Poorly Covered by MedDRA**
> Devansh Verma and Kalyanaraman Pattabiraman
> School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India

Pharmacovigilance pipelines normalise patient-reported adverse drug events (ADEs) to MedDRA. When no
MedDRA concept fits, normalisation still returns the nearest one. GhostCluster addresses the
complementary task of *terminology-gap triage*: it ranks patient expressions by how poorly MedDRA covers
them, using a label-free consensus of semantic distances to all MedDRA Preferred Terms (PTs) and Lowest
Level Terms (LLTs), and groups the most distant expressions into *Ghost Clusters* for human review.

This repository reproduces every number, table and figure in the manuscript from the original data
sources. No data are included; see [data/README.md](data/README.md).

## Method in brief

| Component | Implementation |
|---|---|
| Terminology | MedDRA 24.0: 24,820 PTs and 58,471 LLTs |
| Encoders | S-PubMedBERT-MS-MARCO, SapBERT, BioLORD-2023, all-mpnet-base-v2 |
| Gap score (per encoder) | 1 − mean of the five largest cosine similarities to PT and LLT names |
| Consensus gap (primary) | mean over encoders of the percentile rank of the gap score; no labels, no fitted parameters |
| Hybrid (secondary) | supervised logistic model on length, lexical gap and two encoders' top-1 gaps, cross-validated |
| Ghost Clusters | Ward clustering with about 20 expressions per cluster; the quartile of clusters most distant from MedDRA (HDBSCAN after UMAP for comparison) |
| Evaluation | human no-concept labels in CADEC (MedDRA `CONCEPT_LESS`) and PsyTAR (no exact SNOMED CT concept); baselines, length and repetition controls, supervised transfer, concept-removal simulation; document-level bootstrap confidence intervals |

## Repository structure

```
ghostcluster/            Python package
  config.py              paths, encoders and fixed settings
  data.py                loaders for MedDRA (terms and hierarchy), CADEC v2, PsyTAR and SMM4H-2017 Task 3
  encoders.py            sentence encoders (L2-normalised embeddings)
  scoring.py             gap scores, consensus gap, concept-level gap, lexical gap
  stats.py               AUROC/AUPRC with document-level bootstrap intervals, paired differences
  clustering.py          Ward clustering, HDBSCAN after UMAP, Ghost Cluster selection
scripts/
  01_embed.py            embed term names and corpus expressions
  02_evaluate.py         coverage statistics, human-label evaluation, mapping-difficulty proxy
  03_clusters.py         clustering configuration, Ghost Cluster content, cluster-informed scoring
  04_robustness.py       triage effectiveness, review units and stability
  05_sensitivity.py      length, score definition, baselines, supervised transfer, operating points, label checks
  06_hidden_concepts.py  concept-removal simulation (PTs, HLTs, HLGTs) on SMM4H-2017 and CADEC
  07_additional_analyses.py  OOD scores, BM25 grid, drug transfer, learning curves, batch dependence,
                         spelling normalisation, operating-point intervals, density strata
  08_figures_tables.py   figures and LaTeX tables of the manuscript and Supplementary Information
  run_all.py             runs steps 01-08 in order
annotations/
  ghost_cluster_assessment.csv   authors' gap-type assessment of the most distant Ward Ghost Clusters
data/README.md           how to obtain and place the data
```

## Installation

Python 3.12 was used. Install PyTorch for your platform (https://pytorch.org), then

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for step 01 (embedding); all other steps run on a CPU.

## Reproducing the results

1. Obtain the data and place it as described in [data/README.md](data/README.md).
2. Run

   ```bash
   python scripts/run_all.py
   ```

Results are written to `outputs/`. The figures (`outputs/figures/`) and tables (`outputs/tables/`)
correspond to the manuscript as follows.

| Output | Manuscript |
|---|---|
| `tables/tab_coverage.tex` | Table 3 |
| `tables/tab_human.tex` | Table 4 |
| `tables/tab_operating.tex` | Table 5 |
| `tables/tab_controls.tex` | Table 6 |
| `tables/tab_supervised.tex` | Table 7 |
| `tables/tab_hidden.tex` | Table 8 |
| `tables/tab_triage.tex` | Table 9 |
| `tables/tab_ghosts.tex` | Table 10 |
| `tables/supp_tables.tex` | Supplementary Tables S1-S20 |
| `figures/fig_roc_pr.pdf` | Figure 2 |
| `figures/fig_learning_curve.pdf` | Figure 3 |
| `figures/fig_review_effort.pdf` | Figure 4 |
| `figures/fig_auprc_forest.pdf` | Supplementary Figure S1 |

### Expected key results

A correct run reproduces these values (`outputs/results_human_labels.csv`, `outputs/results_supervised.csv`):

| Check | CADEC | PsyTAR |
|---|---|---|
| Consensus gap, AUROC / AUPRC | 0.879 / 0.312 | 0.716 / 0.225 |
| String match, AUROC / AUPRC | 0.810 / 0.185 | 0.592 / 0.146 |
| Supervised TF-IDF classifier, transferred from the other corpus, AUPRC | 0.069 | 0.161 |

Step 01 stops with an error if MedDRA term names do not retrieve themselves in embedding space, which
guards against embedding term codes instead of names.

Intermediate results (`descriptives.json`, `results_*.csv`, per-mention scores) are saved alongside.
Random seeds are fixed. UMAP is not guaranteed to be bit-identical across library versions or hardware;
the pinned versions in `requirements.txt` were used for the reported results.

On a laptop with an NVIDIA GeForce RTX 5060 (8 GB), the full run takes about one and a half hours.

## Citation

If you use this code, please cite the manuscript (citation details will be added on publication) and
this repository ([CITATION.cff](CITATION.cff)).

## Licence

The code is released under the MIT Licence ([LICENSE](LICENSE)). The data used in the study are
subject to their own licences and are not redistributed here; MedDRA® is a registered trademark of the
International Council for Harmonisation of Technical Requirements for Pharmaceuticals for Human Use (ICH).

## Contact

Kalyanaraman Pattabiraman (corresponding author), pkalyanaraman@vit.ac.in
