"""Step 1 - embed MedDRA term names (PTs and LLTs) and the distinct expressions of each corpus.

Outputs (outputs/):
  meddra_terms.csv                     code, name, level (row order = embedding rows)
  texts_<corpus>.csv                   distinct expressions (row order = embedding rows)
  emb_<encoder>_<meddra|corpus>.npy    L2-normalised embeddings (float32)
The run stops if the embedded terminology texts are not term names or if exact term names do not
retrieve themselves (a guard against embedding identifiers instead of names).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ghostcluster.config import ENCODERS, OUT_DIR, SEED
from ghostcluster.data import load_cadec, load_meddra, load_psytar, load_smm4h, unique_texts
from ghostcluster.encoders import load_encoder


def check_terminology_embeddings(name, M, terms):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(terms), 500, replace=False)
    nearest = (M[idx] @ M.T).argmax(1)
    names = terms.name.str.lower().values
    hit = float(np.mean(names[nearest] == names[idx]))
    print(f"  [{name}] exact term names retrieve themselves: {hit:.1%}")
    if hit < 0.95:
        raise RuntimeError("Terminology embedding check failed")


def main():
    terms, _ = load_meddra()
    terms.to_csv(os.path.join(OUT_DIR, "meddra_terms.csv"), index=False)
    corpora = {"smm4h": unique_texts(load_smm4h().text),
               "cadec": unique_texts(load_cadec().text),
               "psytar": unique_texts(load_psytar().text)}
    for c, texts in corpora.items():
        pd.DataFrame({"text": texts}).to_csv(os.path.join(OUT_DIR, f"texts_{c}.csv"), index=False)
    print("MedDRA:", terms.level.value_counts().to_dict(), "| distinct expressions:",
          {c: len(t) for c, t in corpora.items()})

    for name, (model_id, pooling) in ENCODERS.items():
        print(f"{name}: {model_id}")
        encode = load_encoder(model_id, pooling)
        path = os.path.join(OUT_DIR, f"emb_{name}_meddra.npy")
        if not os.path.exists(path):
            np.save(path, encode(terms.name.tolist()))
        check_terminology_embeddings(name, np.load(path), terms)
        for c, texts in corpora.items():
            path = os.path.join(OUT_DIR, f"emb_{name}_{c}.npy")
            if not os.path.exists(path):
                np.save(path, encode(texts))
        del encode


if __name__ == "__main__":
    main()
