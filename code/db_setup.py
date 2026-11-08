"""Build and query the project database (SQLite). Run from code/: python db_setup.py -> ../data/upi.db
Tables: panel (1,598 member-months), entity_crosswalk (raw name -> harmonized name -> entity), file_log (one row per source file/month).
View: v_bank_month (bank rows only, with derived fields)."""
import sqlite3, json, pandas as pd, numpy as np, os
src = pd.read_csv("../data/processed/upi_remitter_top50_panel.csv"); path = "../data/upi.db"
if os.path.exists(path): os.remove(path)
con = sqlite3.connect(path); cur = con.cursor()
cur.executescript("""
CREATE TABLE panel (month TEXT NOT NULL, rank INTEGER, bank_raw TEXT, bank TEXT, entity TEXT NOT NULL, is_bank INTEGER NOT NULL,
  volume_mn REAL NOT NULL, approved_pct REAL, bd_pct REAL, td_pct REAL NOT NULL, debit_reversal_mn REAL, debit_reversal_success_pct REAL,
  source_file TEXT, month_source TEXT, n_member_rows INTEGER, PRIMARY KEY (month, entity));
CREATE TABLE entity_crosswalk (bank_raw TEXT PRIMARY KEY, bank TEXT, entity TEXT, is_bank INTEGER);
CREATE TABLE file_log (source_file TEXT PRIMARY KEY, month TEXT, month_source TEXT, member_rows INTEGER);
""")
cols = ["month", "rank", "bank_raw", "bank", "entity", "is_bank", "volume_mn", "approved_pct", "bd_pct", "td_pct", "debit_reversal_mn", "debit_reversal_success_pct", "source_file", "month_source", "n_member_rows"]
p = src[cols].copy(); p["is_bank"] = p.is_bank.astype(int)
cur.executemany(f"INSERT INTO panel VALUES ({','.join('?'*len(cols))})", [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in r) for r in p.itertuples(index=False)])
cw = src[["bank_raw", "bank", "entity", "is_bank"]].drop_duplicates("bank_raw").copy(); cw["is_bank"] = cw.is_bank.astype(int)
cur.executemany("INSERT INTO entity_crosswalk VALUES (?,?,?,?)", list(cw.itertuples(index=False)))
fl = src.groupby("source_file").agg(month=("month", "first"), month_source=("month_source", "first"), member_rows=("bank", "size")).reset_index()
cur.executemany("INSERT INTO file_log VALUES (?,?,?,?)", list(fl.itertuples(index=False)))
cur.executescript("""
CREATE INDEX ix_panel_entity ON panel(entity); CREATE INDEX ix_panel_month ON panel(month);
CREATE VIEW v_bank_month AS SELECT month, entity, volume_mn, approved_pct, bd_pct, td_pct FROM panel WHERE is_bank = 1;""")
con.commit()
# integrity checks and the queries used to read the data
checks = dict(rows=cur.execute("SELECT COUNT(*) FROM panel").fetchone()[0], bank_rows=cur.execute("SELECT COUNT(*) FROM v_bank_month").fetchone()[0],
              months=cur.execute("SELECT COUNT(DISTINCT month) FROM panel").fetchone()[0], entities=cur.execute("SELECT COUNT(DISTINCT entity) FROM panel").fetchone()[0],
              files=cur.execute("SELECT COUNT(*) FROM file_log").fetchone()[0],
              max_pct_dev=cur.execute("SELECT MAX(ABS(approved_pct+bd_pct+td_pct-100)) FROM panel").fetchone()[0],
              dup_keys=cur.execute("SELECT COUNT(*) FROM (SELECT month, entity FROM panel GROUP BY month, entity HAVING COUNT(*)>1)").fetchone()[0])
q = """SELECT month, SUM(volume_mn)/1000.0 AS vol_bn, SUM(volume_mn*td_pct)/SUM(volume_mn) AS td_weighted
       FROM v_bank_month GROUP BY month ORDER BY month"""
sysq = pd.read_sql(q, con); checks["sql_first_last"] = [{"month": sysq.month.iloc[0], "vol_bn": round(float(sysq.vol_bn.iloc[0]), 3), "td_weighted": round(float(sysq.td_weighted.iloc[0]), 3)}, {"month": sysq.month.iloc[-1], "vol_bn": round(float(sysq.vol_bn.iloc[-1]), 3), "td_weighted": round(float(sysq.td_weighted.iloc[-1]), 3)}]
df = pd.read_sql("SELECT * FROM v_bank_month", con); checks["pandas_from_sql_rows"] = len(df)
json.dump(checks, open("db_checks.json", "w"), indent=1, default=float); print(checks); con.close()
