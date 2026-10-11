"""Sample-size and descriptive calculations for the UPI technical-decline capstone.
Reads upi_remitter_top50_panel.csv; prints every number used in the synopsis and writes sample_size_results.json."""
import json, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats

Z_A, Z_B = stats.norm.ppf(0.975), stats.norm.ppf(0.80)       # alpha=0.05 two-sided, power=0.80
K = Z_A + Z_B                                                 # ~2.80 (Gelman, Hill & Vehtari)
res = {"z_alpha": Z_A, "z_beta": Z_B, "K": K}

df = pd.read_csv("../data/processed/upi_remitter_top50_panel.csv")
d = df[df.is_bank].copy()
months = sorted(d.month.unique()); midx = {m: i for i, m in enumerate(months)}
d["t"] = d.month.map(midx)
d["p"] = (d.td_pct / 100).clip(lower=0.000025)               # 0.00% placeholders set to half the reporting resolution
d["logit"] = np.log(d.p / (1 - d.p)); d["lnV"] = np.log(d.volume_mn)
res["panel"] = dict(rows_all=len(df), rows_banks=len(d), months=len(months), entities=d.entity.nunique(),
                    entities_all_months=int((d.groupby("entity").month.nunique() == len(months)).sum()))

# ---------------- RQ1: load -> TD, panel with bank + month fixed effects ----------------
X = pd.get_dummies(d[["entity", "month"]], drop_first=True, dtype=float)
X.insert(0, "lnV", d.lnV.values); X = sm.add_constant(X)
fit = sm.OLS(d.logit.values, X).fit()
# within-variation of ln V after removing bank and month effects
Xfe = sm.add_constant(pd.get_dummies(d[["entity", "month"]], drop_first=True, dtype=float))
sd_x = sm.OLS(d.lnV.values, Xfe).fit().resid.std(ddof=1)
sigma_e = np.sqrt(fit.scale)
beta_min = 0.60                                               # 10% more volume -> ~5.9% change in TD odds (exp(0.6*ln1.1)-1)
n_power = (K * sigma_e / (beta_min * sd_x)) ** 2
k_sub, k_fe = 4, (d.entity.nunique() - 1) + (len(months) - 1)
n_green = 50 + 8 * (k_sub + k_fe)
mde_beta = K * sigma_e / (sd_x * np.sqrt(len(d)))
res["RQ1"] = dict(sigma_e=sigma_e, sd_x_within=sd_x, beta_min=beta_min, n_power=n_power, k_sub=k_sub, k_fe=k_fe,
                  n_green=n_green, N=int(np.ceil(max(n_power, n_green))),
                  prelim_beta_lnV=float(fit.params["lnV"]), prelim_se=float(fit.bse["lnV"]), mde_beta_available=float(mde_beta), available=len(d))

# ---------------- RQ2: interrupted time series on monthly volume-weighted TD ----------------
d["tdv"] = d.volume_mn * d.td_pct / 100
m = d.groupby("month").apply(lambda g: pd.Series(dict(td=g.tdv.sum() / g.volume_mn.sum() * 100,
        bd=(g.volume_mn * g.bd_pct / 100).sum() / g.volume_mn.sum() * 100, vol=g.volume_mn.sum()))).reset_index()
m["t"] = m.month.map(midx)
BREAK = "2025-08"; t0 = midx[BREAK]
m["post"] = (m.t >= t0).astype(int); m["t_post"] = (m.t - t0) * m.post
Xi = sm.add_constant(m[["t", "post", "t_post"]]); its = sm.OLS(m.td, Xi).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
sig = np.sqrt(sm.OLS(m.td, Xi).fit().scale)
rho = float(np.corrcoef(sm.OLS(m.td, Xi).fit().resid[:-1], sm.OLS(m.td, Xi).fit().resid[1:])[0, 1])
n1, n2 = int((m.t < t0).sum()), int((m.t >= t0).sum())
infl = (1 + max(rho, 0)) / (1 - max(rho, 0))
delta = 0.10                                                  # 0.10 percentage points of TD
def se_level(n_pre, n_post, sigma_adj):
    """SE of the level change at the break; primary model = level change with a common pre-existing linear trend."""
    t = np.arange(n_pre + n_post); post = (t >= n_pre).astype(float)
    Xd = np.column_stack([np.ones(len(t)), t, post]); cov = np.linalg.inv(Xd.T @ Xd)
    return sigma_adj * np.sqrt(cov[2, 2])
sig_adj = sig * np.sqrt(infl)
se_avail = se_level(n1, n2, sig_adj)
Z95 = stats.norm.ppf(0.975)
W = 0.25                                                      # target 95% CI half-width for the level change (pp of TD)
T_ci = next(T for T in range(12, 300) if Z95 * se_level(round(0.6 * T), T - round(0.6 * T), sig_adj) <= W)
mde = K * se_avail
n_total_power = T_ci
delta = W
res["RQ2"] = dict(break_month=BREAK, n_pre=n1, n_post=n2, sigma=float(sig), rho=rho, infl=infl, ci_target=W,
                  se_level=float(se_avail), ci_half_width_available=float(Z95 * se_avail), n_total_ci=int(T_ci),
                  rule_of_thumb_total=24, mde_pp=float(mde),
                  td_pre_mean=float(m[m.t < t0].td.mean()), td_post_mean=float(m[m.t >= t0].td.mean()),
                  bd_pre_mean=float(m[m.t < t0].bd.mean()), bd_post_mean=float(m[m.t >= t0].bd.mean()),
                  prelim_level_change=float(its.params["post"]), prelim_level_se=float(its.bse["post"]))
res["RQ2"]["N"] = max(res["RQ2"]["n_total_ci"], 24)

# ---------------- RQ3: calendar peaks (Fisher-z correlation, as in course guidance) ----------------
def fisher_n(r): return (K / (0.5 * np.log((1 + r) / (1 - r)))) ** 2 + 3
peak = d.month.str[5:].isin(["03"]) | (d.month.str[5:].isin(["10", "11"]))
res["RQ3"] = dict(r=0.20, n_fisher=float(fisher_n(0.20)), N=int(np.ceil(fisher_n(0.20))),
                  peak_bank_months=int(peak.sum()), march_bank_months=int(d.month.str.endswith("-03").sum()),
                  festive_bank_months=int(d.month.str[5:].isin(["10", "11"]).sum()),
                  n_r30=float(fisher_n(0.30)), n_r25=float(fisher_n(0.25)))

# ---------------- RQ4: spikes, episodes, persistence ----------------
d = d.sort_values(["entity", "t"]).reset_index(drop=True)
med = d.groupby("entity").td_pct.transform("median")
mad = d.groupby("entity").td_pct.transform(lambda x: (x - x.median()).abs().median()) * 1.4826
d["rz"] = (d.td_pct - med) / mad.replace(0, np.nan)
d["spike"] = ((d.rz > 3) & (d.td_pct >= 1.0)).astype(int)
episodes = []
for e, g in d.groupby("entity"):
    run = 0
    for _, r in g.iterrows():
        if r.spike: run += 1
        elif run: episodes.append((e, run, 1)); run = 0
    if run: episodes.append((e, run, 0))                      # still ongoing at end of data (censored)
ep = pd.DataFrame(episodes, columns=["entity", "dur", "recovered"])
train = d[d.month <= "2025-12"]; test = d[d.month > "2025-12"]
# AR(1) persistence on logit TD (within-bank, demeaned), pooled
d["dm"] = d.logit - d.groupby("entity").logit.transform("mean")
d["lag"] = d.groupby("entity").dm.shift(1)
a = d.dropna(subset=["lag"]); rho_ar = float((a.dm * a.lag).sum() / (a.lag ** 2).sum())
# variance decomposition (one-way random effects on logit TD)
import statsmodels.formula.api as smf
mm = smf.mixedlm("logit ~ 1", d, groups=d["entity"]).fit()
s_b = float(mm.cov_re.iloc[0, 0]); s_w = float(mm.scale); icc = s_b / (s_b + s_w)
k_cox, epv = 3, 10
res["RQ4"] = dict(spike_rule="robust z>3 (median/MAD within bank) and TD>=1.0%", spike_months=int(d.spike.sum()),
                  banks_with_spike=int(d[d.spike == 1].entity.nunique()), episodes=len(ep), episodes_recovered=int(ep.recovered.sum()),
                  median_duration=float(ep.dur.median()), mean_duration=float(ep.dur.mean()), max_duration=int(ep.dur.max()),
                  one_month_share=float((ep.dur == 1).mean()), cox_k=k_cox, cox_min_events=k_cox * epv,
                  train_spikes=int(train.spike.sum()), test_spikes=int(test.spike.sum()),
                  ml_features=8, ml_min_positive=8 * 10, total_spikes=int(d.spike.sum()), rho_ar1=rho_ar,
                  half_life=float(np.log(0.5) / np.log(rho_ar)) if 0 < rho_ar < 1 else None,
                  icc=icc, sigma_between=s_b, sigma_within=s_w, banks_for_variance=int(d.entity.nunique()))
res["RQ4"]["N"] = max(30, k_cox * epv)

# ---------------- ML benchmark / naive baseline (calculation example for metrics) ----------------
d["lagTD"] = d.groupby("entity").td_pct.shift(1)
t26 = d[(d.month > "2025-12") & d.lagTD.notna()]
err = t26.td_pct - t26.lagTD
res["naive_baseline"] = dict(n=int(len(t26)), rmse_pp=float(np.sqrt((err ** 2).mean())), mae_pp=float(err.abs().mean()),
                             test_months="2026-01 to 2026-08")
sp = d[(d.month > "2025-12")].copy(); sp["lagspike"] = d.groupby("entity").spike.shift(1).loc[sp.index]
sp = sp.dropna(subset=["lagspike"]); tp = int(((sp.lagspike == 1) & (sp.spike == 1)).sum())
res["naive_spike"] = dict(n=int(len(sp)), pos=int(sp.spike.sum()), precision=tp / max(int((sp.lagspike == 1).sum()), 1),
                          recall=tp / max(int(sp.spike.sum()), 1))

# ---------------- final N ----------------
Ns = {k: res[k]["N"] for k in ["RQ1", "RQ2", "RQ3", "RQ4"]}
res["final_N"] = dict(by_rq=Ns, max=max(Ns.values()), available_bank_months=len(d), available_months=len(months))
json.dump(res, open("sample_size_results.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))
