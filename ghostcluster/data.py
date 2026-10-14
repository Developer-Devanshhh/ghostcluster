"""Loaders for MedDRA, CADEC v2, PsyTAR and SMM4H-2017 Task 3.

None of these resources is redistributed with this repository; data/README.md explains how to
obtain them and where to place them.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from .config import CADEC_DIR, CURRENT_LLTS_ONLY, MEDDRA_DIR, OUT_DIR, PSYTAR_XLSX, SMM4H_DIR

HEADER_MARKER = "Extracted ADR"   # header line present in one SMM4H-2017 normalisation file


def norm(text):
    """Lower-case, keep letters and digits, collapse whitespace (matching and de-duplication only)."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def unique_texts(series):
    """Distinct raw strings in order of first appearance (the units that are embedded)."""
    return list(dict.fromkeys(series.astype(str).tolist()))


# ---------------------------------------------------------------------------------------------
# MedDRA
# ---------------------------------------------------------------------------------------------
def _meddra_from_ascii(asc_dir):
    """Official MSSO distribution: pt.asc and llt.asc ('$'-separated)."""
    def read(name):
        rows = []
        with open(os.path.join(asc_dir, name), encoding="utf-8", errors="replace") as f:
            for line in f:
                p = line.rstrip("\r\n").split("$")
                if len(p) >= 3 and p[0].strip().isdigit():
                    rows.append(p)
        return rows
    pts = read("pt.asc")
    llts = read("llt.asc")
    pt = pd.DataFrame({"code": [int(p[0]) for p in pts], "name": [p[1].strip() for p in pts], "level": "PT"})
    llt_all = pd.DataFrame({"code": [int(p[0]) for p in llts], "name": [p[1].strip() for p in llts],
                            "pt": [int(p[2]) for p in llts],
                            "current": [p[9].strip().upper() != "N" if len(p) > 9 else True for p in llts]})
    llt = llt_all[llt_all.code != llt_all.pt]          # llt.asc repeats every PT as its own LLT
    if CURRENT_LLTS_ONLY:                              # field 10 of llt.asc is the LLT currency flag (Y/N)
        llt = llt[llt.current]
    terms = pd.concat([pt, llt[["code", "name"]].assign(level="LLT")], ignore_index=True)
    return terms, llt[["code", "pt"]].rename(columns={"code": "llt"}).reset_index(drop=True)


def _meddra_from_xlsx(path):
    """Excel export with sheets _ID2NAME (Number, Name), _ID2HIERARCHY (ID, HIERARCHY) and _BROADER (ID, BROADER ID).

    This export carries no LLT currency flag, so current and non-current LLTs are both returned.
    """
    names = pd.read_excel(path, sheet_name="_ID2NAME", engine="openpyxl")
    hier = pd.read_excel(path, sheet_name="_ID2HIERARCHY", engine="openpyxl")
    broader = pd.read_excel(path, sheet_name="_BROADER", engine="openpyxl")
    terms = names.merge(hier, left_on="Number", right_on="ID")[["Number", "Name", "HIERARCHY"]]
    terms.columns = ["code", "name", "level"]
    terms = terms[terms.level.isin(["PT", "LLT"])].reset_index(drop=True)
    level = dict(zip(terms.code, terms.level))
    links = broader[(broader.ID.map(level) == "LLT") & (broader["BROADER ID"].map(level) == "PT")]
    llt2pt = links[["ID", "BROADER ID"]].rename(columns={"ID": "llt", "BROADER ID": "pt"}).reset_index(drop=True)
    terms["name"] = terms.name.astype(str).str.strip()
    return terms, llt2pt


def load_meddra():
    """Return (terms, llt2pt).

    terms:  one row per Preferred Term (PT) and Lowest Level Term (LLT) with columns code, name, level.
    llt2pt: mapping from each LLT code to its parent PT code.
    """
    cache = os.path.join(OUT_DIR, "meddra_terms.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache), pd.read_pickle(cache.replace(".pkl", "_llt2pt.pkl"))
    asc = [d for d in glob.glob(os.path.join(MEDDRA_DIR, "**", "llt.asc"), recursive=True)]
    xlsx = glob.glob(os.path.join(MEDDRA_DIR, "*.xlsx"))
    if asc:
        terms, llt2pt = _meddra_from_ascii(os.path.dirname(asc[0]))
    elif xlsx:
        terms, llt2pt = _meddra_from_xlsx(xlsx[0])
    else:
        raise FileNotFoundError(f"No MedDRA files found in {MEDDRA_DIR}; see data/README.md")
    terms = terms.sort_values(["level", "code"], ascending=[False, True]).reset_index(drop=True)  # PTs first
    assert not terms.name.str.fullmatch(r"\d+").any(), "MedDRA term names must be text, not codes"
    terms.to_pickle(cache)
    llt2pt.to_pickle(cache.replace(".pkl", "_llt2pt.pkl"))
    return terms, llt2pt


def load_meddra_hierarchy():
    """PT -> HLT -> HLGT links as a DataFrame with columns pt, hlt, hlgt (MedDRA is multi-axial)."""
    asc = glob.glob(os.path.join(MEDDRA_DIR, "**", "mdhier.asc"), recursive=True)
    if asc:
        rows = [line.split("$") for line in open(asc[0], encoding="utf-8", errors="replace")]
        return pd.DataFrame({"pt": [int(r[0]) for r in rows], "hlt": [int(r[1]) for r in rows],
                             "hlgt": [int(r[2]) for r in rows]}).drop_duplicates()
    xlsx = glob.glob(os.path.join(MEDDRA_DIR, "*.xlsx"))[0]
    hier = pd.read_excel(xlsx, sheet_name="_ID2HIERARCHY", engine="openpyxl")
    level = dict(zip(hier.ID, hier.HIERARCHY.str.replace("HGLT", "HLGT")))
    b = pd.read_excel(xlsx, sheet_name="_BROADER", engine="openpyxl")
    b = b.assign(a=b.ID.map(level), c=b["BROADER ID"].map(level))
    pt_hlt = b[(b.a == "PT") & (b.c == "HLT")][["ID", "BROADER ID"]].set_axis(["pt", "hlt"], axis=1)
    hlt_hlgt = b[(b.a == "HLT") & (b.c == "HLGT")][["ID", "BROADER ID"]].set_axis(["hlt", "hlgt"], axis=1)
    return pt_hlt.merge(hlt_hlgt, on="hlt").drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------------------------
def _cadec_root():
    hits = glob.glob(os.path.join(CADEC_DIR, "**", "meddra"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"CADEC v2 'meddra' folder not found under {CADEC_DIR}; see data/README.md")
    return os.path.dirname(hits[0])


def load_cadec():
    """CADEC v2 ADR spans from the MedDRA annotation layer.

    nil = 1 when the annotators marked the span CONCEPT_LESS (no MedDRA concept fits); gold is the
    MedDRA code of spans normalised to a single concept (NaN otherwise).
    Only spans whose entity type in the original layer is ADR are kept.
    """
    root = _cadec_root()
    etype = {}
    for f in glob.glob(os.path.join(root, "original", "*.ann")):
        pid = os.path.basename(f)[:-4]
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3 and p[0].startswith("T"):
                    etype[(pid, norm(p[2]))] = p[1].split()[0]
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "meddra", "*.ann"))):
        pid = os.path.basename(f)[:-4]
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                codes = p[1].split()[0] if "+" not in p[1] else ""
                rows.append(dict(post_id=pid, text=p[2].strip(), nil=int("CONCEPT_LESS" in p[1]),
                                 gold=int(codes) if codes.isdigit() else np.nan,
                                 etype=etype.get((pid, norm(p[2])), "?")))
    df = pd.DataFrame(rows)
    return df[df.etype == "ADR"].drop(columns="etype").reset_index(drop=True)


def load_psytar():
    """PsyTAR ADR mentions (sheet ADR_Mapped); nil = 1 when the annotators gave no SNOMED CT code."""
    d = pd.read_excel(PSYTAR_XLSX, sheet_name="ADR_Mapped")
    out = pd.DataFrame({"review_id": d.drug_id.astype(str), "text": d.ADRs.astype(str).str.strip()})
    out["nil"] = (~d["SNOMED-CT"].astype(str).str.lower().str.contains("snomedct", na=False)).astype(int)
    return out


def load_smm4h():
    """SMM4H-2017 Task 3 normalisation data (all files, sorted by name), header line removed."""
    files = sorted(glob.glob(os.path.join(SMM4H_DIR, "**", "task_3_normalization_*.txt"), recursive=True))
    if not files:
        raise FileNotFoundError(f"SMM4H-2017 Task 3 files not found under {SMM4H_DIR}; see data/README.md")
    rows = []
    for f in files:
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                p = line.strip().split("\t")
                if len(p) >= 3:
                    rows.append(dict(id=p[0], text=p[1], gold=p[2]))
    d = pd.DataFrame(rows)
    d = d[~d.text.str.contains(HEADER_MARKER, case=False)].reset_index(drop=True)
    d["gold"] = pd.to_numeric(d.gold, errors="coerce")
    return d


def gold_pt(smm4h, terms, llt2pt):
    """Map SMM4H gold codes to PT codes (LLT codes -> parent PT; other levels -> NaN)."""
    level = dict(zip(terms.code, terms.level))
    parent = dict(zip(llt2pt.llt, llt2pt.pt))
    return np.array([c if level.get(c) == "PT" else parent.get(c, np.nan) for c in smm4h.gold], dtype=float)
