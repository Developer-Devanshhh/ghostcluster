"""Paths, encoders and fixed settings shared by all analysis scripts."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data (not distributed with this repository; see data/README.md).
# Set GHOSTCLUSTER_DATA to use a different location.
DATA_DIR = os.environ.get("GHOSTCLUSTER_DATA", os.path.join(ROOT, "data"))
MEDDRA_DIR = os.path.join(DATA_DIR, "meddra")
CADEC_DIR = os.path.join(DATA_DIR, "cadec")
PSYTAR_XLSX = os.path.join(DATA_DIR, "psytar", "PsyTAR_dataset.xlsx")
SMM4H_DIR = os.path.join(DATA_DIR, "smm4h")

# Generated outputs (embeddings, scores, results, figures, tables).
OUT_DIR = os.environ.get("GHOSTCLUSTER_OUT", os.path.join(ROOT, "outputs"))
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

SEED = 42
N_BOOTSTRAP = 1000

# name: (Hugging Face model id, pooling). "st" = sentence-transformers pooling, "cls" = [CLS] vector.
ENCODERS = {
    "spubmedbert": ("pritamdeka/S-PubMedBert-MS-MARCO", "st"),
    "sapbert": ("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "cls"),
    "biolord": ("FremyCompany/BioLORD-2023", "st"),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "st"),
}
# Encoders inside the primary (hybrid) configuration. Fixed before the other encoders were evaluated.
HYBRID_ENCODERS = ["spubmedbert", "sapbert"]
# Encoder whose embeddings are clustered.
CLUSTER_ENCODER = "spubmedbert"

ENCODER_NAMES = {"spubmedbert": "S-PubMedBERT", "sapbert": "SapBERT",
                 "biolord": "BioLORD-2023", "mpnet": "all-mpnet-base-v2"}

# Clustering settings (HDBSCAN after UMAP); see the manuscript, Section 3.6.
UMAP_PARAMS = dict(n_components=5, n_neighbors=15, min_dist=0.0)
HDBSCAN_PARAMS = dict(min_cluster_size=7, min_samples=7, cluster_selection_method="eom")
HDBSCAN_GRID = dict(min_cluster_size=[7, 10, 15, 20], min_samples=[5, 7, 10], cluster_selection_method=["eom", "leaf"])
GHOST_QUANTILE = 0.75      # Ghost Clusters: clusters whose mean gap score exceeds this quantile
MIN_CLUSTER_SIZE_GHOST = 3

for _d in (OUT_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(_d, exist_ok=True)
