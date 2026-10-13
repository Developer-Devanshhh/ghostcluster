# Data

No data are distributed with this repository. All four resources are available from their original
providers under their own terms. Place them under `data/` as shown below, or set the environment
variable `GHOSTCLUSTER_DATA` to another directory with the same layout.

```
data/
  meddra/      MedDRA distribution (see 1)
  cadec/       CADEC v2 (any folder level that contains the `original/` and `meddra/` sub-folders)
  psytar/      PsyTAR_dataset.xlsx
  smm4h/       task_3_normalization_*.txt (five files)
```

## 1. MedDRA

MedDRA is licensed by the MedDRA Maintenance and Support Services Organization (MSSO):
https://www.meddra.org/how-to-get/subscription. Free licences are available to academic institutions.
The study used **MedDRA version 24.0 (March 2021), English**.

Two formats are accepted in `data/meddra/`:

* the official ASCII distribution, i.e. a folder containing `pt.asc` and `llt.asc` (for example the
  `MedAscii` folder of the release), or
* an Excel workbook with sheets `_ID2NAME` (`Number`, `Name`), `_ID2HIERARCHY` (`ID`, `HIERARCHY`) and
  `_BROADER` (`ID`, `BROADER ID`), which is the export used by the authors.

Both formats yield the same 24,820 PTs and 58,471 LLTs for version 24.0.

## 2. CADEC v2

Karimi S, Metke-Jimenez A, Kemp M, Wang C. CADEC: A corpus of adverse drug event annotations.
*Journal of Biomedical Informatics* 55, 73-81 (2015).
Download from the CSIRO Data Access Portal: https://doi.org/10.4225/08/570FB102BDAD2
Extract the archive into `data/cadec/`. The loader uses `original/*.ann` (entity types) and
`meddra/*.ann` (MedDRA codes and `CONCEPT_LESS` annotations).

## 3. PsyTAR

Zolnoori M et al. A systematic approach for developing a corpus of patient reported adverse drug events:
a case study for SSRI and SNRI medications. *Journal of Biomedical Informatics* 90, 103091 (2019).
The corpus is distributed by its authors subject to the AskaPatient terms of use. Place
`PsyTAR_dataset.xlsx` in `data/psytar/`; the sheet `ADR_Mapped` is used.

## 4. SMM4H-2017 Task 3

Sarker A et al. Data and systems for medication-related text classification and concept normalization
from Twitter: insights from the Social Media Mining for Health (SMM4H)-2017 shared task.
*Journal of the American Medical Informatics Association* 25, 1274-1283 (2018).
Download from Mendeley Data: https://doi.org/10.17632/rxwfb3tysd.2 and copy the five files
`task_3_normalization_*.txt` from `subtask3` into `data/smm4h/`.
