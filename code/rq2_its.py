"""RQ2: interrupted time series for the August 2025 API controls (five specifications, placebo breaks, BD placebo).
Run from code/:  python rq2_its.py   -> writes rq2_results.json"""
import json, numpy as np, pandas as pd, statsmodels.api as sm
raw = pd.read_csv("../data/processed/upi_remitter_top50_panel.csv")
d = raw[raw.is_bank].copy()
mon = d.groupby("month").apply(lambda g: pd.Series(dict(td=(g.volume_mn*g.td_pct).sum()/g.volume_mn.sum(),
                               bd=(g.volume_mn*g.bd_pct).sum()/g.volume_mn.sum()))).reset_index()
mon["t"] = np.arange(len(mon)); B = int(mon.index[mon.month == "2025-08"][0])
def fit(df, y, cols, brk, lag=2, log=False):
    X = df[["t"]].copy() if "t" in cols else pd.DataFrame(index=df.index)
    X["post"] = (df.t >= brk).astype(float)
    X = sm.add_constant(X); yy = np.log(df[y]) if log else df[y]
    m = sm.OLS(yy.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": lag}, use_t=True)
    j = list(X.columns).index("post"); ci = m.conf_int()[j]
    est, lo, hi = m.params[j], ci[0], ci[1]
    if log: est, lo, hi = [(np.exp(v) - 1) * 100 for v in (est, lo, hi)]
    return dict(est=float(est), lo=float(lo), hi=float(hi), p=float(m.pvalues[j]))
specs = {"A": (["post"], False), "B": (["t", "post"], False), "D": (["post"], True), "E": (["t", "post"], True)}
out = {"series_n": len(mon), "break_index": B}
for k, (c, lg) in specs.items():
    out[k] = fit(mon, "td", c, B, log=lg)
w = mon[mon.t >= B - 12]; out["C"] = fit(w, "td", ["post"], B)
cands = list(range(6, len(mon) - 5))   # 6 months on each side -> 21 candidates
out["candidates"] = [mon.month[i] for i in cands]
for k, (c, lg) in specs.items():
    plc = {mon.month[i]: fit(mon, "td", c, i, log=lg)["est"] for i in cands}
    order = sorted(plc, key=lambda m: -abs(plc[m]))
    out[k]["rank"] = order.index("2025-08") + 1; out[k]["placebo"] = plc; out[k]["top3"] = order[:3]
out["BD_trend"] = fit(mon, "bd", ["t", "post"], B); out["BD_notrend"] = fit(mon, "bd", ["post"], B)
rule = all(np.sign(out[k]["est"]) == np.sign(out["A"]["est"]) for k in ["A", "B", "C", "D", "E"]) and out["B"]["rank"] <= 2 and out["A"]["rank"] <= 2
out["rule_met"] = bool(rule)
sd = mon.td.std(ddof=1)
json.dump(out, open("rq2_results.json", "w"), indent=1)
for k in "ABCDE": print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in out[k].items() if a not in ("placebo",)})
print("BD", out["BD_trend"], out["BD_notrend"], "rule", rule)
