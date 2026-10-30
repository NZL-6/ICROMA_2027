# Supplementary material

The PDF accompanies *Indicator-Guided Variable Fixing for Train Timetable Rescheduling*.

- **S1:** Names and operational definitions of the 191 computable, nonconstant indicator items.
- **S2:** Solver, prediction-plus-solver and complete online computation times, including Strategy-2 reference acquisition and separately reported offline costs.
- **S3:** All 6,402 estimable association responses for 4,191 numerical forms, plus input-selection frequencies. Results retain their signs, uncertainty intervals and adjusted significance values; no significance filter is applied.
- **S4:** Names, definitions and availability of the remaining 76 candidate items. Together, S1 and S4 cover all 267 candidate items. Constant or unavailable measurements are not assigned zero correlations.

The companion `indicator_catalog_complete.csv` and `.json` provide the full catalogue and its recorded availability. Join `catalog_id` in `../results/reports/indicator_associations.csv` to the catalogue's `catalog_id` field to obtain the full-precision numerical results. Multiple rows can belong to one indicator because numerical forms, decision types, strategy expansions and response definitions differ. The catalogue also retains the original candidate definitions as provenance; those preliminary definitions may describe earlier pilot conditions. Current availability follows the audited experiment.

## Recompile the document

From the repository root, run:

```text
python supplementary/compile_supplementary.py
```

This compiles only the LaTeX document and does not run training or optimization. It requires XeLaTeX and the packages used in `source/main.tex`. Times New Roman is used when installed; otherwise the open TeX Gyre Termes font is used. Proprietary font files are not bundled. The checked PDF uses Times New Roman.

The main manuscript refers to Supplementary Sections S1--S3. The public repository is <https://github.com/NZL-6/ICROMA_2027>, and the manuscript's supplementary-material statement uses this address.
