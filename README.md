# ms_upi_capstone
MS Capstone - UPI Problem Statement

# UPI Top-50 Remitter panel: build notes
Source: NPCI UPI Ecosystem Statistics, "Top 50 Member Performance - Remitter" (monthly exports, manual download).
Files read: 33. Output: upi_remitter_top50_panel.csv (1,598 rows, 32 months, Jan 2024 - Aug 2026, no gaps).
Rebuild: python3 combine_remitter.py   (see file_log.csv for per-file month/title audit)

## Coverage
All 32 months Jan 2024 - Aug 2026 present. Each month is fed by exactly one file (see month_audit.csv).

## Uniqueness checks (run on every build)
- Content fingerprint per file: identical content under two labels -> duplicate dropped.
- One source file per month; one month per file; no duplicate (month, bank) rows.
- Month-level content hashes all distinct; max share of identical bank volumes between any two months = 2%.

## Fixes applied
1. Percent formats: Feb-Jul 2025 files store % as fractions (0.92); others as text ("92%"). All normalised to 0-100.
   Check: Approved% + BD% + TD% = 100 +/- 0.01 on every row; no nulls.
2. The first file named "2025-Jan" (60e14b47...) was a mislabeled download: title Dec'25, content identical to the
   "2025-Dec" file. Dropped as a duplicate. The re-downloaded Jan 2025 file (Jan-Remitter_2) is genuine.
3. File "2026-Aug" carries the title Jul'26 but its data differ from the "2026-Jul" file (SBI 6,846 vs 6,622 Mn;
   total 26.75 vs 25.90 bn). Treated as Aug 2026 by filename and flagged (month_source = "filename (title mismatch)").
   VERIFY on the NPCI portal that this is genuinely August.
4. Bank names: 130+ raw spellings harmonised to 63 names (`bank`; raw kept in `bank_raw`).
   "AU Small Finance Bank (Erstwhile Fincare)" mapped to Fincare.
5. Slice Small Finance Bank is listed twice in Mar and Jun 2026; combined (volume-summed, volume-weighted rates,
   n_member_rows = 2). Those months have 49 rows.
6. `entity` merges renames (ASSUMPTION, confirm): Andhra Pradesh Grameena Vikas Bank -> Telangana Grameena Bank.
7. Non-bank members kept but flagged is_bank = False: credit-card product lines (Axis, HDFC, ICICI), SBI Cards and
   Payment Services, One MobiKwik Systems, Tri O Tech Solutions (judgement from names; confirm).

## Open items
- 70 of 1,598 rows have TD = 0.00% (true zero vs rounding/placeholder).
- Beneficiary-side files not yet used.

