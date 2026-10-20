"""Interim analysis: cleaning counts, EDA, preliminary models for RQ1-RQ4 and the ML benchmark.
Run from the code/ folder:  python interim_analysis.py
Reads ../data/processed/upi_remitter_top50_panel.csv; writes figures to ../outputs/ and interim_results.json."""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
from scipy import stats
import xgboost as xgb
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
OUT = "../outputs/"
res = {}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.labelcolor": "#222222", "xtick.color": "#333333",
                     "ytick.color": "#333333", "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 200})
BLUE, ORANGE, GREY, DARK = "#1f5a8a", "#c8641a", "#8a8a8a", "#222222"

# ------------------------------------------------------------------ load
raw = pd.read_csv("../data/processed/upi_remitter_top50_panel.csv")
months = sorted(raw.month.unique()); midx = {m: i for i, m in enumerate(months)}
raw["t"] = raw.month.map(midx)
res["panel"] = dict(rows=len(raw), cols=raw.shape[1], months=len(months), first=months[0], last=months[-1],
                    raw_names=int(raw.bank_raw.nunique()), banks=int(raw.bank.nunique()), entities=int(raw.entity.nunique()),
                    nonbank_rows=int((~raw.is_bank).sum()), bank_rows=int(raw.is_bank.sum()))

# ------------------------------------------------------------------ cleaning log numbers
n = len(raw)
res["cleaning"] = dict(
    n_rows=n,
    missing_debit_rev=int(raw.debit_reversal_mn.isna().sum()), missing_debit_succ=int(raw.debit_reversal_success_pct.isna().sum()),
    missing_core=int(raw[["volume_mn", "approved_pct", "bd_pct", "td_pct"]].isna().sum().sum()),
    zero_td=int((raw.td_pct == 0).sum()),
    dup_month_bank=int(raw.duplicated(["month", "bank"]).sum()),
    two_member_rows=int((raw.n_member_rows > 1).sum()),
    pct_sum_max_dev=float((raw.pct_sum - 100).abs().max()),
    title_mismatch_rows=int((raw.month_source != "sheet title").sum()),
    n_months_49=int((raw.groupby("month").size() == 49).sum()),
)
d = raw[raw.is_bank].copy().sort_values(["entity", "t"]).reset_index(drop=True)
d["p"] = (d.td_pct / 100).clip(lower=0.000025)
d["logit"] = np.log(d.p / (1 - d.p)); d["lnV"] = np.log(d.volume_mn)
d["dlnV"] = d.groupby("entity").lnV.diff()
d["gap"] = d.groupby("entity").t.diff()
d.loc[d.gap != 1, "dlnV"] = np.nan
d["cummax_prev"] = d.groupby("entity").lnV.transform(lambda x: x.cummax().shift(1))
d["newpeak"] = ((d.lnV > d.cummax_prev) & d.cummax_prev.notna()).astype(int)
d["mon"] = d.month.str[5:].astype(int)
d["peak"] = d.mon.isin([3, 10, 11]).astype(int)
d["post"] = (d.month >= "2025-08").astype(int)

# outlier flags
med = d.groupby("entity").td_pct.transform("median")
mad = d.groupby("entity").td_pct.transform(lambda x: (x - x.median()).abs().median()) * 1.4826
d["rz"] = (d.td_pct - med) / mad.replace(0, np.nan)
d["spike"] = ((d.rz > 3) & (d.td_pct >= 1.0)).astype(int)
q1, q3 = d.td_pct.quantile([.25, .75]); iqr = q3 - q1
res["cleaning"].update(dict(iqr_upper=float(q3 + 1.5 * iqr), iqr_outliers=int((d.td_pct > q3 + 1.5 * iqr).sum()),
                            n_bank=len(d), spike_months=int(d.spike.sum()), spike_banks=int(d[d.spike == 1].entity.nunique()),
                            td_max=float(d.td_pct.max()), td_p99=float(d.td_pct.quantile(.99)),
                            vol_max_share=float(d.volume_mn.max() / d.groupby("month").volume_mn.sum().iloc[0])))
res["cleaning"]["td_max_row"] = d.loc[d.td_pct.idxmax(), ["entity", "month", "td_pct", "volume_mn"]].to_dict()

# bank groups (rule-based, documented)
PUBLIC = ["State Bank of India", "Bank of Baroda", "Punjab National Bank", "Canara Bank", "Union Bank of India", "Bank of India",
          "Indian Bank", "Indian Overseas Bank", "Central Bank of India", "UCO Bank", "Bank of Maharashtra", "Punjab & Sind Bank"]
PRIVATE = ["HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank", "IndusInd Bank", "YES Bank", "IDFC FIRST Bank",
           "Federal Bank", "RBL Bank", "Bandhan Bank", "South Indian Bank", "Karur Vysya Bank", "City Union Bank",
           "Karnataka Bank", "Jammu & Kashmir Bank", "Tamilnad Mercantile Bank", "IDBI Bank"]
def grp(e):
    if e in PUBLIC: return "Public sector"
    if e in PRIVATE: return "Private sector"
    if "Small Finance" in e: return "Small finance"
    if "Payments Bank" in e: return "Payments bank"
    if "Gramin" in e or "Grameena" in e: return "Regional rural"
    return "Other"
d["group"] = d.entity.map(grp)
res["groups"] = d.groupby("group").entity.nunique().to_dict()

# ------------------------------------------------------------------ descriptive statistics
def desc(s):
    return dict(n=int(s.count()), mean=float(s.mean()), sd=float(s.std()), min=float(s.min()), p50=float(s.median()), p99=float(s.quantile(.99)), max=float(s.max()))
res["desc"] = {k: desc(d[k]) for k in ["volume_mn", "approved_pct", "bd_pct", "td_pct"]}
res["desc_all_td_mean"] = float(d.td_pct.mean())
mon = d.groupby("month").apply(lambda g: pd.Series(dict(
    vol=g.volume_mn.sum(), td=(g.volume_mn * g.td_pct).sum() / g.volume_mn.sum(),
    bd=(g.volume_mn * g.bd_pct).sum() / g.volume_mn.sum(), appr=(g.volume_mn * g.approved_pct).sum() / g.volume_mn.sum()))).reset_index()
mon["t"] = mon.month.map(midx)
res["system"] = dict(vol_first=float(mon.vol.iloc[0]), vol_last=float(mon.vol.iloc[-1]), growth_x=float(mon.vol.iloc[-1] / mon.vol.iloc[0]),
                     cagr_month=float((mon.vol.iloc[-1] / mon.vol.iloc[0]) ** (1 / (len(mon) - 1)) - 1),
                     td_first=float(mon.td.iloc[0]), td_last=float(mon.td.iloc[-1]), td_min=float(mon.td.min()), td_min_month=mon.loc[mon.td.idxmin(), "month"],
                     td_max=float(mon.td.max()), td_max_month=mon.loc[mon.td.idxmax(), "month"],
                     bd_first=float(mon.bd.iloc[0]), bd_last=float(mon.bd.iloc[-1]),
                     td_2024=float(mon[mon.month < "2025-01"].td.mean()), td_post=float(mon[mon.month >= "2025-08"].td.mean()),
                     corr_vol_td=float(mon[["vol", "td"]].corr().iloc[0, 1]), corr_vol_bd=float(mon[["vol", "bd"]].corr().iloc[0, 1]))
by_group = d.groupby("group").apply(lambda g: pd.Series(dict(banks=g.entity.nunique(), rows=len(g), mean_td=g.td_pct.mean(), median_td=g.td_pct.median(),
                                                            p95_td=g.td_pct.quantile(.95), spike_rate=g.spike.mean() * 100,
                                                            vol_share=g.volume_mn.sum() / d.volume_mn.sum() * 100))).reset_index()
res["by_group"] = by_group.round(3).to_dict(orient="records")
bank = d.groupby("entity").agg(vol=("volume_mn", "mean"), td=("td_pct", "mean"), tdmed=("td_pct", "median"), n=("td_pct", "size"), sp=("spike", "sum"), group=("group", "first")).reset_index()
res["bank_corr_logvol_td_spearman"] = float(stats.spearmanr(np.log(bank.vol), bank.tdmed)[0])
res["bank_corr_n"] = int(len(bank))
big = bank.sort_values("vol", ascending=False).head(10)
res["top10_median_td"] = float(big.tdmed.mean()); res["rest_median_td"] = float(bank.sort_values("vol").head(len(bank) - 10).tdmed.mean())
res["top10_share_vol"] = float(d[d.entity.isin(big.entity)].volume_mn.sum() / d.volume_mn.sum())
# within-bank variation
res["within_between_td"] = dict(total_sd=float(d.td_pct.std()), between_sd=float(d.groupby("entity").td_pct.mean().std()),
                                within_sd=float((d.td_pct - d.groupby("entity").td_pct.transform("mean")).std()))
# seasonality: bank-demeaned TD by calendar month
d["td_dm"] = d.td_pct - d.groupby("entity").td_pct.transform("mean")
season = d.groupby("mon").td_dm.agg(["mean", "sem", "count"]).reset_index()
res["season"] = season.round(4).to_dict(orient="records")
# volume growth vs TD change
d["dTD"] = d.groupby("entity").td_pct.diff(); d.loc[d.gap != 1, "dTD"] = np.nan
g = d.dropna(subset=["dlnV", "dTD"])
g_trim = g[(g.dTD.abs() < g.dTD.abs().quantile(.99))]
res["growth_td"] = dict(n=int(len(g)), pearson=float(g[["dlnV", "dTD"]].corr().iloc[0, 1]), spearman=float(stats.spearmanr(g.dlnV, g.dTD)[0]),
                        spearman_p=float(stats.spearmanr(g.dlnV, g.dTD)[1]),
                        big_jump_n=int((g.dlnV > 0.2).sum()), big_jump_mean_dtd=float(g[g.dlnV > 0.2].dTD.mean()),
                        mid_mean_dtd=float(g[(g.dlnV.abs() <= 0.2)].dTD.mean()), drop_n=int((g.dlnV < -0.2).sum()),
                        drop_mean_dtd=float(g[g.dlnV < -0.2].dTD.mean()), dlnV_sd=float(g.dlnV.std()),
                        newpeak_share=float(d.newpeak.mean()), newpeak_dtd=float(g[g.newpeak == 1].dTD.mean()), nonpeak_dtd=float(g[g.newpeak == 0].dTD.mean()))
# spike clustering by month
spk_month = d.groupby("month").spike.sum().reindex(months).fillna(0).astype(int)
res["spike_by_month"] = spk_month.to_dict()
res["spike_month_top"] = spk_month.sort_values(ascending=False).head(5).to_dict()

# ------------------------------------------------------------------ figures: EDA
def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT + name, bbox_inches="tight", facecolor="white"); plt.close(fig)

xt = list(range(0, len(months), 3)); xl = [months[i] for i in xt]
# Fig 1: system trend
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
ax[0].plot(mon.t, mon.vol / 1000, color=BLUE, lw=1.8); ax[0].set_title("Attempted volume (billion)", fontsize=9, loc="left")
ax[1].plot(mon.t, mon.td, color=ORANGE, lw=1.8); ax[1].set_title("Technical decline rate (%)", fontsize=9, loc="left")
ax[2].plot(mon.t, mon.bd, color=GREY, lw=1.8, ls="--"); ax[2].set_title("Business decline rate (%)", fontsize=9, loc="left")
for a in ax:
    a.set_xticks(xt[::2]); a.set_xticklabels(xl[::2], rotation=45, ha="right", fontsize=7.5)
    a.axvline(midx["2025-08"], color="#444444", lw=0.9, ls=":")
ax[1].text(midx["2025-08"] + .4, mon.td.max() * .97, "Aug 2025", fontsize=7.5)
save(fig, "fig1_system_trends.png")

# Fig 2: distribution of TD
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].hist(d.td_pct.clip(upper=3), bins=60, color=BLUE, alpha=.9); ax[0].set_yscale("log")
ax[0].set_xlabel("Technical decline rate, % (values above 3 shown at 3)"); ax[0].set_ylabel("Bank-months (log scale)")
ax[0].axvline(1.0, color=ORANGE, ls="--", lw=1.2); ax[0].text(1.05, ax[0].get_ylim()[1] * .35, "1.0% spike floor", color=ORANGE, fontsize=8)
order = d.groupby("group").td_pct.median().sort_values().index.tolist()
ax[1].boxplot([d[d.group == g_].td_pct.clip(upper=3) for g_ in order], tick_labels=[o.replace(" ", "\n", 1) for o in order], showfliers=True,
              flierprops=dict(marker="o", markersize=2, markerfacecolor=GREY, markeredgecolor=GREY), medianprops=dict(color=ORANGE, lw=1.6),
              boxprops=dict(color=DARK), whiskerprops=dict(color=DARK), capprops=dict(color=DARK))
ax[1].set_ylabel("TD rate, % (clipped at 3)"); ax[1].tick_params(axis="x", labelsize=7.5)
save(fig, "fig2_td_distribution.png")

# Fig 3: bank heterogeneity scatter
fig, ax = plt.subplots(figsize=(5.4, 3.2))
cols = {"Public sector": BLUE, "Private sector": ORANGE, "Small finance": "#5b8e3e", "Payments bank": "#7b4a9c", "Regional rural": GREY, "Other": "#bbbbbb"}
mk = {"Public sector": "o", "Private sector": "s", "Small finance": "^", "Payments bank": "D", "Regional rural": "v", "Other": "x"}
for gname, gg in bank.groupby("group"):
    ax.scatter(gg.vol, gg.tdmed, s=26, color=cols[gname], marker=mk[gname], label=gname, alpha=.9, lw=.8)
ax.set_xscale("log"); ax.set_xlabel("Mean monthly attempted volume per bank (million, log scale)"); ax.set_ylabel("Median TD rate (%)")
r_ = bank.sort_values("vol", ascending=False).iloc[0]
ax.annotate("State Bank of India", (r_.vol, r_.tdmed), fontsize=7.5, xytext=(-6, 8), textcoords="offset points", ha="right")
ax.legend(fontsize=7.5, frameon=False, ncol=2)
save(fig, "fig3_bank_scatter.png")

# Fig 4: seasonality
fig, ax = plt.subplots(figsize=(5.6, 2.8))
cc = [ORANGE if m in (3, 10, 11) else BLUE for m in season.mon]
ax.bar(season.mon, season["mean"], color=cc, width=.7, yerr=1.96 * season["sem"], error_kw=dict(ecolor="#555555", lw=.8, capsize=2))
ax.axhline(0, color=DARK, lw=.8); ax.set_xticks(range(1, 13)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_ylabel("TD vs bank average (pp)")
save(fig, "fig4_seasonality.png")

# Fig 5: volume growth vs TD change
fig, ax = plt.subplots(figsize=(5.4, 3.0))
gg = g[(g.dTD.abs() < 3) & (g.dlnV.abs() < 0.8)]
ax.scatter(gg.dlnV, gg.dTD, s=7, color=BLUE, alpha=.35, lw=0)
bins = pd.cut(gg.dlnV, [-0.8, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.8])
bm = gg.groupby(bins, observed=True).agg(x=("dlnV", "mean"), y=("dTD", "mean"))
bm["n"] = gg.groupby(bins, observed=True).size().values
ax.plot(bm.x, bm.y, color=ORANGE, lw=2, marker="o", ms=4, label="Mean change in each volume-growth band (n shown)")
for _, r_ in bm.iterrows(): ax.annotate(f"n={int(r_.n)}", (r_.x, r_.y), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=7)
ax.axhline(0, color=DARK, lw=.8); ax.set_xlabel("Month-over-month log growth in volume"); ax.set_ylabel("Change in TD rate (pp)")
ax.legend(fontsize=8, frameon=False, loc="lower right")
save(fig, "fig5_growth_vs_td.png")

# Fig 6: spikes by month
fig, ax = plt.subplots(figsize=(7.0, 2.5))
ax.bar(range(len(months)), spk_month.values, color=[ORANGE if m == "2025-08" else BLUE for m in months], width=.75)
ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=7.5); ax.set_ylabel("Banks in spike")
save(fig, "fig6_spikes_by_month.png")

# ------------------------------------------------------------------ preliminary models
# ---- RQ1: fractional logit, entity + month FE, cluster-robust SE
def frac_fit(df, xcols, fe_month=True, extra=None):
    X = pd.get_dummies(df[["entity"] + (["month"] if fe_month else [])], drop_first=True, dtype=float)
    X = pd.concat([df[xcols].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    if extra is not None: X = pd.concat([X, extra.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    codes = pd.factorize(df.entity)[0]
    m = sm.GLM(df.p.values, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": codes})
    mu = m.predict(X); wt = float((mu * (1 - mu)).mean())
    return m, wt
r1 = d.dropna(subset=["dlnV"]).copy()
r1["lagp"] = r1.groupby("entity").p.shift(1)
rq1 = {}
for name, xc, dd in [("A", ["dlnV", "newpeak"], r1), ("B", ["dlnV", "newpeak", "lagp"], r1.dropna(subset=["lagp"]))]:
    m, wt = frac_fit(dd, xc)
    rq1[name] = dict(n=int(len(dd)), clusters=int(dd.entity.nunique()),
                     **{f"{v}_coef": float(m.params[v]) for v in xc}, **{f"{v}_se": float(m.bse[v]) for v in xc}, **{f"{v}_p": float(m.pvalues[v]) for v in xc},
                     ame_dlnV_pp_per_10pct=float(m.params["dlnV"] * wt * 100 * np.log(1.1)),
                     ame_newpeak_pp=float(m.params["newpeak"] * wt * 100),
                     ci_dlnV=[float(m.conf_int().loc["dlnV", 0]), float(m.conf_int().loc["dlnV", 1])],
                     ci_newpeak=[float(m.conf_int().loc["newpeak", 0]), float(m.conf_int().loc["newpeak", 1])],
                     mean_wt=wt, odds_ratio_dlnV_10pct=float(np.exp(m.params["dlnV"] * np.log(1.1))))
res["rq1"] = rq1

# ---- RQ3: peaks (entity FE + linear trend + Post + Peak + Peak x exposure)
r3 = d.copy(); r3["tt"] = r3.t / 10.0
expo = r3.groupby("entity").lnV.transform("mean"); r3["expo"] = (expo - expo.mean()) / expo.std()
r3["peak_x_expo"] = r3.peak * r3.expo
m3, wt3 = frac_fit(r3, ["peak", "peak_x_expo", "tt", "post"], fe_month=False)
res["rq3"] = dict(n=int(len(r3)), peak_bank_months=int(r3.peak.sum()),
                  peak_coef=float(m3.params["peak"]), peak_se=float(m3.bse["peak"]), peak_p=float(m3.pvalues["peak"]),
                  int_coef=float(m3.params["peak_x_expo"]), int_se=float(m3.bse["peak_x_expo"]), int_p=float(m3.pvalues["peak_x_expo"]),
                  ame_peak_pp=float(m3.params["peak"] * wt3 * 100), ci_peak=[float(x) for x in m3.conf_int().loc["peak"]],
                  raw_peak_mean=float(d[d.peak == 1].td_dm.mean()), raw_other_mean=float(d[d.peak == 0].td_dm.mean()),
                  march_mean=float(d[d.mon == 3].td_dm.mean()))
# March 2025 sensitivity (drop the Mar-2025 outage-adjacent months)
r3b = r3[~r3.month.isin(["2025-03", "2025-04"])]
m3b, wt3b = frac_fit(r3b, ["peak", "peak_x_expo", "tt", "post"], fe_month=False)
res["rq3"]["peak_coef_ex_mar25"] = float(m3b.params["peak"]); res["rq3"]["peak_se_ex_mar25"] = float(m3b.bse["peak"])

# ---- RQ4: ICC, AR(1), episodes, KM, Cox
mm = smf.mixedlm("logit ~ 1", d, groups=d["entity"]).fit()
s_b = float(mm.cov_re.iloc[0, 0]); s_w = float(mm.scale)
d["dm"] = d.logit - d.groupby("entity").logit.transform("mean"); d["lagdm"] = d.groupby("entity").dm.shift(1)
a_ = d.dropna(subset=["lagdm"]); rho = float((a_.dm * a_.lagdm).sum() / (a_.lagdm ** 2).sum())
episodes = []
for e, gg_ in d.groupby("entity"):
    run, start = 0, None
    for _, r in gg_.iterrows():
        if r.spike:
            if run == 0: start = r.month
            run += 1
        elif run:
            episodes.append((e, run, 1, start)); run = 0
    if run: episodes.append((e, run, 0, start))
ep = pd.DataFrame(episodes, columns=["entity", "dur", "event", "start"])
ep["start_peak"] = ep.start.str[5:].astype(int).isin([3, 10, 11]).astype(int)
ep["group"] = ep.entity.map(grp); ep["public"] = (ep.group == "Public sector").astype(int)
bm_ = d.groupby("entity").lnV.mean(); ep["lnV"] = ep.entity.map(bm_); ep["lnV_z"] = (ep.lnV - bm_.mean()) / bm_.std()
km, S = [], 1.0
for tt_ in range(1, int(ep.dur.max()) + 1):
    n_ = int((ep.dur >= tt_).sum()); d_ = int(((ep.dur == tt_) & (ep.event == 1)).sum())
    S *= (1 - d_ / n_) if n_ else 1; km.append(dict(t=tt_, at_risk=n_, events=d_, S=S))
res["km"] = km
try:
    cox = PHReg(ep.dur.values.astype(float), ep[["lnV_z", "start_peak", "public"]].values, status=ep.event.values, ties="efron").fit()
    res["cox"] = dict(names=["lnV_z", "start_peak", "public"], hr=[float(x) for x in np.exp(cox.params)], p=[float(x) for x in cox.pvalues],
                      ci=[[float(a), float(b)] for a, b in np.exp(cox.conf_int())], n=int(len(ep)), events=int(ep.event.sum()))
except Exception as ex:
    res["cox"] = dict(error=str(ex))
res["rq4"] = dict(icc=s_b / (s_b + s_w), s_between=s_b, s_within=s_w, rho=rho, half_life=float(np.log(.5) / np.log(rho)),
                  spike_months=int(d.spike.sum()), banks=int(d[d.spike == 1].entity.nunique()), episodes=int(len(ep)),
                  recovered=int(ep.event.sum()), median=float(ep.dur.median()), mean=float(ep.dur.mean()), max=int(ep.dur.max()),
                  one_month=float((ep.dur == 1).mean()), start_peak_share=float(ep.start_peak.mean()),
                  by_group_mean=ep.groupby("group").dur.mean().round(2).to_dict(), ep_by_group=ep.groupby("group").size().to_dict())
fig, ax = plt.subplots(figsize=(4.2, 2.5))
xs = [0] + [k["t"] for k in km]; ys = [1.0] + [k["S"] for k in km]
ax.step(xs, ys, where="post", color=BLUE, lw=2); ax.set_ylim(0, 1.02); ax.set_xticks(xs)
ax.set_xlabel("Months since spike began"); ax.set_ylabel("Share of spikes still ongoing")
for x_, y_ in zip(xs[1:], ys[1:]): ax.annotate(f"{y_:.2f}", (x_, y_), xytext=(5, 5), textcoords="offset points", fontsize=8)
save(fig, "fig7_km.png")

# ------------------------------------------------------------------ prediction benchmark
d = d.sort_values(["entity", "t"]).reset_index(drop=True)
d["td_l1"] = d.groupby("entity").td_pct.shift(1); d["td_l2"] = d.groupby("entity").td_pct.shift(2)
d["bd_l0"] = d.bd_pct
d["hist_mean"] = d.groupby("entity").td_pct.transform(lambda x: x.expanding().mean())
d["next_td"] = d.groupby("entity").td_pct.shift(-1); d["next_t"] = d.groupby("entity").t.shift(-1)
d["next_spike"] = d.groupby("entity").spike.shift(-1)
d["next_month"] = d.groupby("entity").month.shift(-1)
M = d[(d.next_t == d.t + 1) & d.dlnV.notna() & d.td_l1.notna()].copy()
M["nm"] = M.next_month.str[5:].astype(int)
M["sin_m"] = np.sin(2 * np.pi * M.nm / 12); M["cos_m"] = np.cos(2 * np.pi * M.nm / 12); M["peak_next"] = M.nm.isin([3, 10, 11]).astype(int)
for gname in ["Public sector", "Private sector", "Small finance", "Payments bank", "Regional rural"]:
    M["g_" + gname.split()[0]] = (M.group == gname).astype(int)
FEATS = ["td_pct", "td_l1", "bd_l0", "lnV", "dlnV", "newpeak", "hist_mean", "sin_m", "cos_m", "peak_next"] + [c for c in M.columns if c.startswith("g_")]
res["features"] = FEATS
test_months = [m for m in months if m >= "2026-01"]
rows = []
def metrics(y, yh): return float(np.sqrt(np.mean((y - yh) ** 2))), float(np.mean(np.abs(y - yh)))
preds = {k: [] for k in ["naive", "histmean", "glm", "xgb", "xgbc"]}
idx_all = []
for m in test_months:
    tr = M[M.next_month < m]; te = M[M.next_month == m]
    if te.empty: continue
    idx_all += list(te.index)
    preds["naive"].append(te.td_pct.values)
    preds["histmean"].append(te.hist_mean.values)
    # fractional logit with group dummies
    Xtr = sm.add_constant(tr[FEATS].astype(float)); Xte = sm.add_constant(te[FEATS].astype(float), has_constant="add")
    g1 = sm.GLM((tr.next_td / 100).clip(0, 0.5).values, Xtr, family=sm.families.Binomial()).fit()
    preds["glm"].append(g1.predict(Xte).values * 100)
    xr = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=2, verbosity=0)
    xr.fit(tr[FEATS], tr.next_td.clip(upper=10)); preds["xgb"].append(xr.predict(te[FEATS]))
    pos = tr.next_spike.sum(); neg = len(tr) - pos
    xc = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                           scale_pos_weight=neg / max(pos, 1), n_jobs=2, verbosity=0)
    xc.fit(tr[FEATS], tr.next_spike); preds["xgbc"].append(xc.predict_proba(te[FEATS])[:, 1])
T = M.loc[idx_all].copy()
for k in preds: T["p_" + k] = np.concatenate(preds[k])
y = T.next_td.values; ys = T.next_spike.values.astype(int)
perf = {}
for k in ["naive", "histmean", "glm", "xgb"]:
    r_, m_ = metrics(y, T["p_" + k].values)
    # robust version: RMSE excluding the single largest-error bank-month
    perf[k] = dict(rmse=r_, mae=m_)
    score = T["p_" + k].values
    perf[k]["pr_auc"] = float(average_precision_score(ys, score)) if ys.sum() else None
    pred_sp = (score >= 1.0).astype(int) if k != "naive" else T.spike.values.astype(int)
    tp = int(((pred_sp == 1) & (ys == 1)).sum())
    perf[k].update(precision=tp / max(int(pred_sp.sum()), 1), recall=tp / max(int(ys.sum()), 1), flagged=int(pred_sp.sum()))
sc = T.p_xgbc.values; pred_sp = (sc >= 0.5).astype(int); tp = int(((pred_sp == 1) & (ys == 1)).sum())
perf["xgb_clf"] = dict(pr_auc=float(average_precision_score(ys, sc)), precision=tp / max(int(pred_sp.sum()), 1), recall=tp / max(int(ys.sum()), 1), flagged=int(pred_sp.sum()))
perf["n_test"] = int(len(T)); perf["test_spikes"] = int(ys.sum()); perf["base_rate"] = float(ys.mean())
# error sensitivity: share of squared error from top-3 errors in naive and xgb
for k in ["naive", "xgb"]:
    e2 = (y - T["p_" + k].values) ** 2; perf[k]["top3_sq_share"] = float(np.sort(e2)[-3:].sum() / e2.sum())
    keep = np.argsort(e2)[:-3]; perf[k]["rmse_ex_top3"] = float(np.sqrt(e2[keep].mean()))
# feature importance (last fit, regression)
imp = pd.Series(xr.feature_importances_, index=FEATS).sort_values(ascending=False)
perf["importance"] = imp.round(3).to_dict()
res["perf"] = perf
res["train_spikes_end"] = int(M[M.next_month < "2026-01"].next_spike.sum()); res["train_n_end"] = int((M.next_month < "2026-01").sum())

# bar chart of RMSE / MAE
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
names = ["Naive\n(last month)", "Bank\nhistory mean", "Fractional\nlogit", "XGBoost"]; ks = ["naive", "histmean", "glm", "xgb"]
cb = [GREY, GREY, BLUE, ORANGE]
for a, met, lab in [(ax[0], "rmse", "RMSE (pp)"), (ax[1], "mae", "MAE (pp)")]:
    vals = [perf[k][met] for k in ks]; a.bar(range(4), vals, color=cb, width=.65)
    a.set_xticks(range(4)); a.set_xticklabels(names, fontsize=7.5); a.set_ylabel(lab)
    for i, v in enumerate(vals): a.text(i, v + max(vals) * .02, f"{v:.3f}", ha="center", fontsize=8)
save(fig, "fig8_model_compare.png")

# ------------------------------------------------------------------ RQ2 (already estimated in rq2_robustness.py)
res["rq2"] = json.load(open("rq2_results.json"))

json.dump(res, open("interim_results.json", "w"), indent=1, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
print("done")
