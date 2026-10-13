# GhostCluster

Code for the manuscript

> **Ghost Clusters: Detecting Patient-Reported Adverse Drug Event Expressions That Fall Outside Controlled Medical Terminologies**
> Devansh Verma and Kalyanaraman Pattabiraman
> School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India

Pharmacovigilance pipelines normalise patient-reported adverse drug events (ADEs) to MedDRA. When no
MedDRA concept fits, normalisation still returns the nearest one. GhostCluster addresses the
complementary task of *terminology-gap detection*: it scores each patient expression by its semantic
distance to all MedDRA Preferred Terms (PTs) and Lowest Level Terms (LLTs), combines this score with
lexical evidence, and groups the most distant expressions into *Ghost Clusters* for human review.

This repository reproduces every number, table and figure in the manuscript from the original data
sources. No data are included; see [data/README.md](data/README.md).

## Method in brief

| Component | Implementation |
|---|---|
| Terminology | MedDRA 24.0: 24,820 PTs and 58,471 LLTs |
| Encoders | S-PubMedBERT-MS-MARCO, SapBERT, BioLORD-2023, all-mpnet-base-v2 |
| Semantic gap score | 1 − mean of the five largest cosine similarities to PT and LLT names |
| Lexical gap score | 1 − best token-sort similarity (`rapidfuzz`) to any term name |
| Hybrid score (primary) | logistic model on length, lexical gap and the S-PubMedBERT and SapBERT gap scores, cross-validated by document |
| Ghost Clusters | HDBSCAN after UMAP, or Ward clustering; the quartile of clusters most distant from MedDRA |
| Evaluation | human no-concept labels in CADEC (MedDRA `CONCEPT_LESS`) and PsyTAR (no SNOMED CT code); hidden-concept simulation on SMM4H-2017; document-level bootstrap confidence intervals |

## Repository structure

```
ghostcluster/            Python package
  config.py              paths, encoders and fixed settings
  data.py                loaders for MedDRA, CADEC v2, PsyTAR and SMM4H-2017 Task 3
  encoders.py            sentence encoders (L2-normalised embeddings)
  scoring.py             semantic and lexical gap scores
  stats.py               AUROC/AUPRC with document-level bootstrap intervals
  clustering.py          HDBSCAN after UMAP, Ward clustering, Ghost Cluster selection
  crf.py                 lexicon-based relevance filter (explored, not part of the framework)
scripts/
  01_embed.py            embed term names and corpus expressions
  02_evaluate.py         coverage statistics, human-label evaluation, SMM4H analyses
  03_clusters.py         clustering configuration, cluster-level scoring, Ghost Cluster content
  04_robustness.py       triage effectiveness and stability across seeds, encoders and Ward clustering
  05_figures_tables.py   figures and LaTeX tables of the manuscript and Supplementary Information
  run_all.py             runs steps 01-05 in order
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
| `tables/tab_hidden.tex` | Table 5 |
| `tables/tab_triage.tex` | Table 6 |
| `tables/tab_ghosts.tex` | Table 7 |
| `tables/supp_tables.tex` | Supplementary Tables S1-S3 |
| `figures/fig_roc_pr.pdf` | Figure 2 |
| `figures/fig_auprc_forest.pdf` | Figure 3 |

Intermediate results (`descriptives.json`, `results_*.csv`, per-mention scores) are saved alongside.
Random seeds are fixed. UMAP is not guaranteed to be bit-identical across library versions or hardware;
the pinned versions in `requirements.txt` were used for the reported results.

On a laptop with an NVIDIA GeForce RTX 5060 (8 GB), the full run takes about 30 minutes.

## Citation

If you use this code, please cite the manuscript (citation details will be added on publication) and
this repository ([CITATION.cff](CITATION.cff)).

## Licence

The code is released under the MIT Licence ([LICENSE](LICENSE)). The data used in the study are
subject to their own licences and are not redistributed here; MedDRA® is a registered trademark of the
International Council for Harmonisation of Technical Requirements for Pharmaceuticals for Human Use (ICH).

## Contact

Kalyanaraman Pattabiraman (corresponding author), pkalyanaraman@vit.ac.in
