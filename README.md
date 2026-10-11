# UPI Technical Declines: Volume Shifts and Regulatory Interventions

QM 640 Data Analytics Capstone, Walsh College (MS in AIML, via Great Learning)
Author: Karthikeyan S | Mentor: Sharat Srivatsa | Term 3, October 2026

## Research aim
Estimate how sudden volume shifts and regulatory interventions affect technical decline (TD) rates in UPI, to help NPCI and banks decide where to add capacity and which API controls to enforce. The full synopsis is in `docs/QM640_Synopsis_Final_Karthikeyan_S.pdf`.

| RQ | Question | Method (script) |
|----|----------|-----------------|
| RQ1 | Do month-on-month volume surges raise TD share? | Fractional (quasi-binomial) logit, bank and month fixed effects, bank-clustered SEs (planned analysis) |
| RQ2 | Did the 2025 NPCI API controls change TD? | Interrupted time series, five specifications, placebo breaks (`code/rq2_robustness.py`) |
| RQ3 | Do peak months (March year-end, Oct-Nov festive) raise TD? | Event study with Peak x exposure interaction (planned analysis) |
| RQ4 | How persistent are TD spikes? | ICC, AR(1), Kaplan-Meier and Cox (planned analysis) |

Sample size for all four questions: `code/sample_size.py` (final N = 1,522 bank-months against 1,555 available).

## Data
- Source: NPCI UPI Ecosystem Statistics, "Top 50 Member Performance", Remitter side, Jan 2024 to Aug 2026. https://www.npci.org.in/product/ecosystem-statistics/upi
- Public statistics, downloaded manually (NPCI does not permit automated access). No personal data.
- Panel: 1,598 bank-month rows, 32 months. Approved % + business decline % + technical decline % = 100 within 0.01.
- Cleaning decisions and caveats: `README_data_notes.md`.

## Folder structure
```
ms_upi_capstone/
  README.md                  this file
  README_data_notes.md       cleaning decisions, caveats
  data/
    raw/                     33 original NPCI monthly xlsx files (unaltered)
    processed/
      upi_remitter_top50_panel.csv   combined analysis panel
      file_log.csv           one row per raw file (provenance)
      month_audit.csv        per-month integrity checks
      entity_crosswalk.csv   bank-name mapping used in RQ2 sensitivity
  code/
    combine_remitter.py      raw xlsx -> panel
    sample_size.py           sample-size calculations (RQ1-RQ4) -> sample_size_results.json
    rq2_robustness.py        RQ2 specifications, placebo test -> rq2_results.json, fig2.png
  docs/
    QM640_Synopsis_Final_Karthikeyan_S.pdf
```

## How to reproduce
Requires Python 3.10+ and: `pandas numpy scipy statsmodels openpyxl matplotlib`
(tested with pandas 3.0, statsmodels 0.15).

```bash
git clone https://github.com/karthick2613-git/ms_upi_capstone.git
cd ms_upi_capstone
pip install pandas numpy scipy statsmodels openpyxl matplotlib

cd code                       # scripts use relative paths, run them from code/
python combine_remitter.py    # 1. rebuilds data/processed/ (panel, file_log, month_audit)
python sample_size.py         # 2. sample-size table -> sample_size_results.json
python rq2_robustness.py      # 3. RQ2 results -> rq2_results.json, fig2.png, entity_crosswalk.csv
```

Step 1 can be skipped: the processed panel is already included. Steps 2 and 3 read `data/processed/upi_remitter_top50_panel.csv`.

## Reproducibility notes
- Raw files are kept unaltered; all cleaning is in code and logged.
- The build script validates: one file per month, no duplicate bank-month rows, percentages sum to 100.
- Known issues (details in `README_data_notes.md`): two raw files mislabeled at source, 130+ bank-name spellings reduced to 63, 70 rows with TD = 0.00% (real zeros or placeholders).
- Not yet used: Beneficiary-side tables and months after Aug 2026.

## Source and licence note
Data © NPCI, used for academic research with attribution. Code by the author for coursework.
