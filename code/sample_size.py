"""Minimum sample size per research question (alpha .05 two-sided, power .80). Run from code/: python sample_size.py"""
import json, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
raw = pd.read_csv("../data/processed/upi_remitter_top50_panel.csv"); months = sorted(raw.month.unique()); midx = {m: i for i, m in enumerate(months)}
d = raw[raw.is_bank].copy(); d["t"] = d.month.map(midx); d = d.sort_values(["entity", "t"]).reset_index(drop=True)
p = (d.td_pct/100).clip(lower=.000025); d["logit"] = np.log(p/(1-p)); d["lnV"] = np.log(d.volume_mn)
d["dlnV"] = d.groupby("entity").lnV.diff(); d.loc[d.groupby("entity").t.diff() != 1, "dlnV"] = np.nan
K = stats.norm.ppf(.975) + stats.norm.ppf(.80); out = dict(K=K)
fe = smf.ols("logit ~ lnV + C(entity) + C(month)", d).fit(); se = float(np.sqrt(fe.scale)); out["sigma_e"] = se
vv = smf.ols("lnV ~ C(entity) + C(month)", d).fit(); sdx = float(vv.resid.std()); out["sdx_level"] = sdx
sdg = float(d.dlnV.std()); out["sdx_growth"] = sdg; out["n_growth_pairs"] = int(d.dlnV.notna().sum())
beta = 0.60; out["rq1_level_n"] = (K*se/(beta*sdx))**2; out["rq1_growth_n"] = (K*se/(beta*sdg))**2
out["detectable_beta_growth"] = K*se/(sdg*np.sqrt(out["n_growth_pairs"])); out["odds_pct_10pct_jump_beta060"] = float(np.exp(beta*np.log(1.1))-1)*100
# RQ2: CI half-width with AR(1)-inflated SE
mon = d.groupby("month").apply(lambda g: (g.volume_mn*g.td_pct).sum()/g.volume_mn.sum()).values
tt = np.arange(len(mon)); r = sm.OLS(mon, sm.add_constant(np.column_stack([tt, (tt >= 19).astype(float)]))).fit()
sig = float(np.sqrt(r.scale)); rho = float(np.corrcoef(r.resid[:-1], r.resid[1:])[0, 1]); infl = np.sqrt((1+rho)/(1-rho))
out.update(rq2_sigma=sig, rq2_rho=rho, rq2_inflation=infl)
def se_at(n):   # SE of the step in a trend-plus-step model for n months with the break after 60% of them, inflated for AR(1)
    t_ = np.arange(n); X = np.column_stack([np.ones(n), t_, (t_ >= round(.6*n)).astype(float)])
    return sig*infl*np.sqrt(np.linalg.inv(X.T @ X)[2, 2])
out["rq2_se_32"] = se_at(32); n2 = next(n for n in range(8, 120) if 1.96*se_at(n) <= 0.25); out["rq2_required_months"] = n2
out["rq2_detectable_pp"] = K*se_at(32)
# RQ3: Fisher z
r_ = .20; n3 = (K/np.arctanh(r_))**2+3; peak_n = int(d.month.str[5:].astype(int).isin([3, 10, 11]).sum()); out.update(rq3_peak_required=n3, rq3_peak_available=peak_n, rq3_bank_months_required=n3/(peak_n/len(d)))
# RQ4a: ICC CI width
rho_icc = 0.557; m = len(d)/d.entity.nunique(); k = d.entity.nunique()
v = 2*(1-rho_icc)**2*(1+(m-1)*rho_icc)**2/(m*(m-1)*(k-1)); out.update(icc_m=m, icc_se=np.sqrt(v), icc_half=1.96*np.sqrt(v), icc_k_needed=k*(1.96*np.sqrt(v)/.10)**2)
# RQ4b: Schoenfeld
dd = K**2/(.25*np.log(2)**2); out.update(rq4b_events=dd, rq4b_bank_months=np.ceil(dd)/(69/len(d)))
json.dump(out, open("sample_size_results.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
