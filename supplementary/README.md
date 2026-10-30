# Supplementary material

The PDF accompanies *Learning to Fix Retiming and Reordering Decisions in Railway Timetable Rescheduling*.

- **S1:** Names and operational definitions of the 191 computable, nonconstant indicator items, followed by the 22 retrieved studies confirmed by the authors as the literature context used to develop the catalogue.
- **S2:** Solver, prediction-plus-solver and complete online computation times, including Strategy-2 reference acquisition and separately reported offline costs.
- **S3:** All 6,402 estimable association responses for 4,191 numerical forms, plus input-selection frequencies. Results retain their signs, uncertainty intervals and adjusted significance values; no significance filter is applied.
- **S4:** Names, definitions and availability of the remaining 76 candidate items. Together, S1 and S4 cover all 267 candidate items. Constant or unavailable measurements are not assigned zero correlations.
- **S5:** STAT and ML architectures, shared input processing, calibration and fixing rules, and the actual regularization settings and ML training lengths used in each round and in the frozen transfer models.

The companion `indicator_catalog_complete.csv` and `.json` provide the full catalogue and its recorded availability. Join `catalog_id` in `../results/reports/indicator_associations.csv` to the catalogue's `catalog_id` field to obtain the full-precision numerical results. Multiple rows can belong to one indicator because numerical forms, decision types, strategy expansions and response definitions differ. The catalogue also retains the original candidate definitions as provenance; those preliminary definitions may describe earlier pilot conditions. Current availability follows the audited experiment.

`retrieved_studies.csv` and `.json` preserve the 22-study inventory, original titles, source links, and the original reading-level records. The inventory does not imply that full texts were available for every study or that every candidate indicator was defined in a source paper. The Chinese-language study uses its publisher's English title in the PDF while retaining the original Chinese metadata in the data files.

## Recompile the document

From the repository root, run:

```text
python supplementary/compile_supplementary.py
```

This compiles only the LaTeX document and does not run training or optimization. It requires XeLaTeX and the packages used in `source/main.tex`. Times New Roman is used when installed; otherwise the open TeX Gyre Termes font is used. Proprietary font files are not bundled. The checked PDF uses Times New Roman.

The main manuscript refers to Supplementary Sections S1--S5. The public repository is <https://github.com/NZL-6/ICROMA_2027>, and the manuscript's supplementary-material statement uses this address.
