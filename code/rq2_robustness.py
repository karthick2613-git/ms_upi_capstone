"""RQ2 robustness + entity-mapping sensitivity. Run from this folder: python3 rq2_robustness.py
Reads ../upi_remitter_top50_panel.csv; writes rq2_results.json, fig2.png, entity_crosswalk.csv."""
import json, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
P = pd.Period; BRK = P("2025-08", "M"); LAGS = 2
d = pd.read_csv("../data/processed/upi_remitter_top50_panel.csv"); b = d[d.is_bank].copy()
b["per"] = pd.PeriodIndex(b.month, freq="M")

def wavg(col):
    s = b.groupby("per").apply(lambda x: np.average(x[col], weights=x.volume_mn), include_groups=False); return s
td, bd = wavg("td_pct"), wavg("bd_pct")

def its(y, brk, trend=True, window=None):
    y = y.copy()
    if window: y = y[(y.index >= brk - window[0]) & (y.index < brk + window[1])]
    t = np.arange(len(y)); post = (y.index >= brk).astype(float)
    X = sm.add_constant(np.column_stack([t, post]) if trend else post)
    r = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": LAGS})
    k = X.shape[1] - 1; bhat, se = r.params[k], r.bse[k]; crit = stats.t.ppf(.975, len(y) - X.shape[1])
    return dict(b=float(bhat), se=float(se), lo=float(bhat - crit * se), hi=float(bhat + crit * se), n=int(len(y)))

specs = {
 "A. Step only (no trend)":            dict(y=td, trend=False, window=None, log=False),
 "B. Linear trend + step (planned)":   dict(y=td, trend=True,  window=None, log=False),
 "C. 12 months before vs after, step": dict(y=td, trend=False, window=(12, 13), log=False),
 "D. Log TD, step only":               dict(y=np.log(td), trend=False, window=None, log=True),
 "E. Log TD, linear trend + step":     dict(y=np.log(td), trend=True,  window=None, log=True),
}
cands = [p for p in td.index if p >= td.index[6] and p <= td.index[-6]]   # >=6 months each side
out = {"specs": {}, "bd_placebo": {}}
for name, s in specs.items():
    est = its(s["y"], BRK, s["trend"], s["window"])
    if s["log"]:
        for k in ("b", "lo", "hi"): est[k + "_pct"] = float((np.exp(est[k]) - 1) * 100)
    # placebo-in-time: same spec at every candidate break (full-series specs only)
    if s["window"] is None:
        allb = {str(c): its(s["y"], c, s["trend"])["b"] for c in cands}
        v = np.array(list(allb.values())); a = allb[str(BRK)]
        est["n_breaks"] = len(v); est["rank_abs"] = int((np.abs(v) >= abs(a)).sum())
        est["n_same_sign_as_aug_larger"] = int(((np.sign(v) == np.sign(a)) & (np.abs(v) >= abs(a))).sum())
        if name.startswith("B"): out["placebo_B"] = allb
        if name.startswith("D"): out["placebo_D"] = allb
    out["specs"][name] = est
for nm, tr in (("step_only", False), ("trend_step", True)):
    out["bd_placebo"][nm] = its(bd, BRK, tr)
out["jun_vs_aug_B"] = {"jun2025": its(td, P("2025-06", "M"), True), "aug2025": its(td, BRK, True)}
out["means"] = dict(pre=float(td[td.index < BRK].mean()), post=float(td[td.index >= BRK].mean()),
                    pre12=float(td[(td.index >= BRK - 12) & (td.index < BRK)].mean()), n_pre=int((td.index < BRK).sum()), n_post=int((td.index >= BRK).sum()))

# ---- panel: entity-mapping sensitivity (fractional logit, entity FE, clustered by entity) ----
b["y"] = b.td_pct.clip(lower=0.0025) / 100
b["t"] = b.per.astype(int) - b.per.astype(int).min(); b["post"] = (b.per >= BRK).astype(int)
def fit(df, f):
    r = smf.glm(f, data=df, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.entity)[0]})
    return dict(coef=float(r.params["post"]), se=float(r.bse["post"]), OR=float(np.exp(r.params["post"])), n=int(r.nobs), clusters=int(df.entity.nunique()))
variants = {"Mapped (as planned)": b,
            "Not mapped (two entities)": b.assign(entity=np.where(b.bank == "Andhra Pradesh Grameena Vikas Bank", "APGVB_separate", b.entity)),
            "Excluded": b[b.entity != "Telangana Grameena Bank"]}
out["entity"] = {k: {"no_trend": fit(v, "y ~ post + C(entity)"), "trend": fit(v, "y ~ t + post + C(entity)")} for k, v in variants.items()}
out["entity_rows"] = int((b.entity == "Telangana Grameena Bank").sum()); out["bank_rows"] = int(len(b))

# ---- exploratory dose-response: Post x pre-period size, month fixed effects ----
pre = b[(b.per >= BRK - 7) & (b.per < BRK)]; size = pre.groupby("entity").volume_mn.mean().apply(np.log)
size = (size - size.mean()) / size.std(); e = b[b.entity.isin(size.index)].copy(); e["z"] = e.entity.map(size); e["mon"] = e.per.astype(str)
r = smf.glm("y ~ post:z + C(entity) + C(mon)", data=e, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.entity)[0]})
k = [i for i in r.params.index if "post" in i][0]
out["dose_size"] = dict(coef=float(r.params[k]), se=float(r.bse[k]), n=int(r.nobs), clusters=int(e.entity.nunique()))

# ---- crosswalk ----
cw = d[d.is_bank].groupby(["entity", "bank"]).agg(first_month=("month", "min"), last_month=("month", "max"), months=("month", "nunique"),
        raw_names=("bank_raw", lambda x: " | ".join(sorted(set(x))))).reset_index()
cw["mapping_type"] = np.where(cw.groupby("entity").bank.transform("nunique") > 1, "ENTITY RENAME (assumption; see note)", "spelling variants only")
cw.to_csv("../data/processed/entity_crosswalk.csv", index=False)

# ---- figure 2: placebo-break estimates, spec B ----
pb = out["placebo_B"]; xs = list(pb.keys()); ys = [pb[k] for k in xs]
plt.rcParams["font.family"] = "serif"; fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.axhline(0, color="gray", lw=.8); ax.plot(range(len(xs)), ys, "k-o", ms=3)
for lab, c, off, ha in (("Jun 2025 (announcement)", "2025-06", (-12, 24), "right"), ("Aug 2025 (enforcement)", "2025-08", (12, 24), "left")):
    i = xs.index(c); ax.plot(i, pb[c], "o", ms=9, mfc="none", mec="k")
    ax.annotate(lab, (i, pb[c]), textcoords="offset points", xytext=off, fontsize=7, ha=ha, arrowprops=dict(arrowstyle="-", lw=.5))
ax.set_ylim(-0.3, 0.3)
ax.set_xticks(range(0, len(xs), 2)); ax.set_xticklabels(xs[::2], rotation=45, fontsize=7); ax.set_ylabel("Estimated level change (pp)"); ax.set_xlabel("Assumed break month"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig2.png", dpi=200)
json.dump(out, open("rq2_results.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ("specs", "bd_placebo", "jun_vs_aug_B", "means", "entity", "entity_rows", "bank_rows", "dose_size")}, indent=1, default=float))
