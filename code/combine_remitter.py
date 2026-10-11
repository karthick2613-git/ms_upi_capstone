"""Combine NPCI UPI Top-50 remitter member performance monthly files into one tidy panel.
Run: python3 combine_remitter.py   (reads the monthly .xlsx exports, writes CSVs in cwd)"""
import glob, re, os, openpyxl, pandas as pd

SRC = "../data/raw/"
MONTHS = {m: i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}

def num(x):
    """number; plain floats <=1 in percent columns are fractions handled by pct()"""
    if x is None: return None
    s = str(x).replace(",", "").replace("%", "").strip()
    return float(s) if s not in ("", "-", "NA") else None

def pct(x):
    """Return percent on 0-100 scale. Text like '95.62%' is already percent; numeric cells are fractions."""
    if x is None: return None
    if isinstance(x, str): return num(x)
    return float(x) * 100.0

# canonical bank names (regex on lowercase raw name, first match wins)
CANON = [
 (r"credit card", None),   # product lines (Axis/HDFC credit card), kept under own label below
 (r"erstwhile fincare", "Fincare Small Finance Bank"), (r"^au small", "AU Small Finance Bank"), (r"^airtel", "Airtel Payments Bank"),
 (r"^andhra pradesh grameena vikas", "Andhra Pradesh Grameena Vikas Bank"),
 (r"^andhra pragathi", "Andhra Pragathi Grameena Bank"),
 (r"^andhra pradesh grameena bank", "Andhra Pradesh Grameena Bank"),
 (r"^axis", "Axis Bank"), (r"^bandhan", "Bandhan Bank"), (r"^bank of baroda", "Bank of Baroda"),
 (r"^bank of india", "Bank of India"), (r"^bank of maharashtra", "Bank of Maharashtra"),
 (r"^baroda u", "Baroda UP Gramin Bank"), (r"^citi", "Citibank"), (r"^chaitanya godavari", "Chaitanya Godavari Grameena Bank"), (r"^one mobikwik", "One MobiKwik Systems"), (r"^sbi cards", "SBI Cards and Payment Services"), (r"^canara", "Canara Bank"),
 (r"^central bank", "Central Bank of India"), (r"^city union", "City Union Bank"),
 (r"^dbs", "DBS Bank India"), (r"^esaf", "ESAF Small Finance Bank"), (r"^equitas", "Equitas Small Finance Bank"),
 (r"^federal", "Federal Bank"), (r"^fincare", "Fincare Small Finance Bank"), (r"^fino", "Fino Payments Bank"),
 (r"^hdfc", "HDFC Bank"), (r"^icici", "ICICI Bank"), (r"^idbi", "IDBI Bank"), (r"^idfc", "IDFC FIRST Bank"),
 (r"^indusind", "IndusInd Bank"), (r"^india post", "India Post Payments Bank"), (r"^indian bank", "Indian Bank"),
 (r"^indian overseas", "Indian Overseas Bank"), (r"^jio", "Jio Payments Bank"),
 (r"^(the )?jammu", "Jammu & Kashmir Bank"), (r"^karnataka bank", "Karnataka Bank"),
 (r"^(karnataka gramin|pragathi krishna)", "Karnataka Gramin Bank"), (r"^karur", "Karur Vysya Bank"),
 (r"^kerala gramin", "Kerala Gramin Bank"), (r"^kotak", "Kotak Mahindra Bank"),
 (r"^maharashtra gramin", "Maharashtra Gramin Bank"), (r"^nsdl", "NSDL Payments Bank"),
 (r"^paytm", "Paytm Payments Bank"), (r"^punjab (&|and) sind", "Punjab & Sind Bank"),
 (r"^punjab national", "Punjab National Bank"), (r"^rbl", "RBL Bank"),
 (r"^rajasthan (marudhara )?gramin", "Rajasthan Marudhara Gramin Bank"), (r"^saraswat", "Saraswat Co-operative Bank"),
 (r"^sarva haryana", "Sarva Haryana Gramin Bank"), (r"^slice", "Slice Small Finance Bank"),
 (r"^(the )?south indian", "South Indian Bank"), (r"^standard chartered", "Standard Chartered Bank"),
 (r"^state bank", "State Bank of India"), (r"^tamilnad", "Tamilnad Mercantile Bank"),
 (r"^telangana", "Telangana Grameena Bank"), (r"^tri o tech", "Tri O Tech Solutions"),
 (r"^uco", "UCO Bank"), (r"^ujjivan", "Ujjivan Small Finance Bank"), (r"^union bank", "Union Bank of India"),
 (r"^uttar pradesh gramin", "Uttar Pradesh Gramin Bank"), (r"^yes bank", "YES Bank"),
]
def canon(raw):
    s = raw.lower().strip()
    if "credit card" in s: return raw.strip()           # product line, not the bank itself
    for pat, name in CANON:
        if name and re.search(pat, s): return name
    raise ValueError("unmapped bank name: " + raw)

rows, log = [], []
for f in sorted(glob.glob(SRC + "*Remitter*.xlsx")):
    name = os.path.basename(f)
    y, m = re.search(r"performance-(\d{4})-([A-Za-z]{3})-", name).groups()
    ws = openpyxl.load_workbook(f, data_only=True).worksheets[0]
    data = list(ws.iter_rows(values_only=True))
    title = data[0][0]
    tm = re.search(r"\((\w{3})'(\d{2})\)", title)
    title_ym = f"20{tm.group(2)}-{MONTHS[tm.group(1)]:02d}" if tm else None
    file_ym = f"{y}-{MONTHS[m]:02d}"
    n = 0
    for r in data[2:]:
        if r[1] in (None, ""): continue
        rows.append(dict(
            month=title_ym, file_month=file_ym, rank=int(float(str(r[0]).replace(",", ""))),
            bank_raw=str(r[1]).strip(), bank=canon(str(r[1]).strip()),
            volume_mn=num(r[2]), approved_pct=pct(r[3]), bd_pct=pct(r[4]), td_pct=pct(r[5]),
            debit_reversal_mn=num(r[6]), debit_reversal_success_pct=pct(r[7]), source_file=name))
        n += 1
    log.append(dict(file_name=name, file_month=file_ym, title_month=title_ym, n_banks=n, title_mismatch=file_ym != title_ym))

df = pd.DataFrame(rows)
df["month_source"] = "sheet title"
# ---- Content fingerprint per source file: catches identical data filed under different month labels ----
import hashlib
fp = {f: hashlib.md5(g.sort_values("rank")[["bank_raw","volume_mn","approved_pct","bd_pct","td_pct"]]
                      .round(4).to_csv(index=False).encode()).hexdigest()
      for f, g in df.groupby("source_file")}
by_fp = {}
for f, h in fp.items(): by_fp.setdefault(h, []).append(f)
fp_dups = {h: fs for h, fs in by_fp.items() if len(fs) > 1}
for h, fs in fp_dups.items(): print("IDENTICAL CONTENT:", [x[:8] + "..." + x.split("performance-")[1] for x in fs])
for h, fs in fp_dups.items():
    keepers = [f for f in fs if next(l for l in log if l["file_name"] == f)["title_mismatch"] is False]
    keep = keepers[0] if keepers else sorted(fs)[0]
    for f in fs:
        if f != keep:
            print(f"DROP (identical content) {f}")
            df = df[df.source_file != f]
log = [l for l in log if l["file_name"] in set(df.source_file)]
# Title/filename mismatches: (a) data identical to the title-month file -> mislabeled duplicate, drop;
# (b) data differs -> portal title is stale; use the filename month and flag it.
key = ["rank", "bank_raw", "volume_mn", "td_pct", "bd_pct"]
for l in [x for x in log if x["title_mismatch"]]:
    f, fm, tm = l["file_name"], l["file_month"], l["title_month"]
    a = df[df.source_file == f][key].reset_index(drop=True)
    others = df[(df.month == tm) & (df.source_file != f)]
    twin = any(a.equals(others[others.source_file == o][key].reset_index(drop=True)) for o in others.source_file.unique())
    if twin:
        print(f"DROP {f}: titled {tm}, identical to the {tm} file (mislabeled download of {fm})")
        df = df[df.source_file != f]
    else:
        print(f"RELABEL {f}: titled {tm} but data differ from the {tm} file -> using filename month {fm} (flagged)")
        df.loc[df.source_file == f, ["month", "month_source"]] = [fm, "filename (title mismatch)"]
# Same member listed twice in a month (e.g. Slice SFB): combine, volume-weighted
def combine(g):
    if len(g) == 1: return g.iloc[0]
    w = g.volume_mn / g.volume_mn.sum(); r = g.iloc[0].copy()
    r["volume_mn"] = g.volume_mn.sum(); r["debit_reversal_mn"] = g.debit_reversal_mn.sum()
    for c in ["approved_pct", "bd_pct", "td_pct", "debit_reversal_success_pct"]:
        r[c] = (g[c] * w).sum() if c != "debit_reversal_success_pct" else (g[c] * g.debit_reversal_mn).sum() / g.debit_reversal_mn.sum()
    r["rank"] = g["rank"].min(); r["bank_raw"] = " + ".join(g.bank_raw)
    return r
df["n_member_rows"] = 1
dups = df.groupby(["month", "bank"]).size(); dups = dups[dups > 1]
for (mth, b), n in dups.items(): print(f"COMBINED {n} listings of {b} in {mth}")
df["_k"] = df.month + "|" + df.bank
cnt = df.groupby("_k").size()
df["n_member_rows"] = df._k.map(cnt)
df = pd.DataFrame([combine(g) for _, g in df.groupby("_k", sort=False)]).drop(columns="_k").reset_index(drop=True)
df["rank"] = df["rank"].astype(int); df["n_member_rows"] = df.n_member_rows.astype(int)
df = df.sort_values(["month", "rank"]).reset_index(drop=True)
# entity continuity (renames) -- ASSUMPTIONS, confirm before use in RQ7
ENTITY = {"Andhra Pradesh Grameena Vikas Bank": "Telangana Grameena Bank"}
df["entity"] = df.bank.replace(ENTITY)
NON_BANK = {"Tri O Tech Solutions", "One MobiKwik Systems", "SBI Cards and Payment Services"}  # non-bank members (judgement from names)
df["is_bank"] = ~(df.bank.str.contains("Credit Card") | df.bank.isin(NON_BANK))
df["pct_sum"] = (df.approved_pct + df.bd_pct + df.td_pct).round(2)
# ---- Integrity checks ----
assert df.groupby("month").source_file.nunique().eq(1).all(), "a month is fed by more than one file"
assert df.groupby("source_file").month.nunique().eq(1).all(), "a file feeds more than one month"
assert not df.duplicated(["month", "bank"]).any(), "duplicate month-bank rows"
mfp = df.groupby("month").apply(lambda g: hashlib.md5(g.sort_values("rank")[["bank_raw","volume_mn","td_pct"]].round(4).to_csv(index=False).encode()).hexdigest())
assert mfp.is_unique, "two months have identical content"
# near-duplicate scan: share of banks with identical volume between any two months
piv = df.pivot(index="bank", columns="month", values="volume_mn").round(3)
cols = list(piv.columns); worst = (0.0, ("", ""))
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        both = piv[[cols[i], cols[j]]].dropna()
        share = (both[cols[i]] == both[cols[j]]).mean() if len(both) else 0
        worst = max(worst, (share, (cols[i], cols[j])))
print(f"Max share of identical bank volumes between any two months: {worst[0]:.0%} {worst[1]}")
assert worst[0] < 0.2, "near-duplicate months"
audit = df.groupby("month").agg(source_file=("source_file", "first"), month_source=("month_source", "first"),
                               rows=("bank", "size"), total_volume_mn=("volume_mn", "sum")).round(2)
audit["content_hash"] = mfp.str[:10]
audit.to_csv("../data/processed/month_audit.csv")
print("INTEGRITY OK:", df.month.nunique(), "months,", len(df), "rows, one source file per month")
df.to_csv("../data/processed/upi_remitter_top50_panel.csv", index=False)
pd.DataFrame(log).sort_values("title_month").to_csv("../data/processed/file_log.csv", index=False)
print(df.shape, df.bank.nunique(), "canonical banks")
