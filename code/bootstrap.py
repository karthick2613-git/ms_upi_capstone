"""Cluster (bank) pairs bootstrap for RQ1 model A and RQ3: percentile 95% intervals and bootstrap p-values (400 resamples per model).
Run from code/: python bootstrap.py -> bootstrap_results.json"""
import json, warnings, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore"); rng = np.random.RandomState(42)
raw = pd.read_csv("../data/processed/upi_remitter_top50_panel.csv"); months = sorted(raw.month.unique()); midx = {m: i for i, m in enumerate(months)}
d = raw[raw.is_bank].copy(); d["t"] = d.month.map(midx); d = d.sort_values(["entity", "t"]).reset_index(drop=True)
d["p"] = (d.td_pct/100).clip(lower=.000025); d["lnV"] = np.log(d.volume_mn); d["dlnV"] = d.groupby("entity").lnV.diff(); d.loc[d.groupby("entity").t.diff() != 1, "dlnV"] = np.nan
cm = d.groupby("entity").lnV.transform(lambda x: x.cummax().shift(1)); d["newpeak"] = ((d.lnV > cm) & cm.notna()).astype(int)
d["peak"] = d.month.str[5:].astype(int).isin([3, 10, 11]).astype(int); d["post"] = (d.month >= "2025-08").astype(int); d["tt"] = d.t/10.0
expo = d.groupby("entity").lnV.transform("mean"); d["expo"] = (expo-expo.mean())/expo.std(); d["peak_x_expo"] = d.peak*d.expo
def fit(df, xc, fe_month):
    X = pd.get_dummies(df[["cl"] + (["month"] if fe_month else [])].astype(str), drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([df[xc].reset_index(drop=True), X.reset_index(drop=True)], axis=1))
    m = sm.GLM(df.p.values, X, family=sm.families.Binomial()).fit(); mu = m.predict(X); return m.params, float((mu*(1-mu)).mean())
def run(df0, xc, fe_month, B=400):
    ents = df0.entity.unique(); df0 = df0.copy(); df0["cl"] = df0.entity
    p0, w0 = fit(df0, xc, fe_month); est0 = {v: p0[v] for v in xc}; bs = {v: [] for v in xc}
    groups = {e: df0[df0.entity == e] for e in ents}
    for b in range(B):
        parts = []
        for k, e in enumerate(rng.choice(ents, len(ents), replace=True)):
            g = groups[e].copy(); g["cl"] = f"{e}_{k}"; parts.append(g)
        db = pd.concat(parts)
        try:
            pb, _ = fit(db, xc, fe_month)
            for v in xc: bs[v].append(pb[v])
        except Exception: pass
    out = {}
    for v in xc:
        a = np.array(bs[v]); out[v] = dict(coef=float(est0[v]), lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)), p=float(min(1, 2*min((a <= 0).mean(), (a >= 0).mean()))), n_boot=int(len(a)),
                                          ame_scale=float(w0*100))
    return out
r1 = d.dropna(subset=["dlnV"])
out = {"rq1_A": run(r1, ["dlnV", "newpeak"], True), "rq3": run(d, ["peak", "peak_x_expo", "tt", "post"], False)}
json.dump(out, open("bootstrap_results.json", "w"), indent=1); print(json.dumps(out, indent=1))
