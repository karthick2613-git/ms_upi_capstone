"""Prediction benchmark: next-month TD (regression) and next-month spike (classification).
Split by target month: train < 2025-01 (initial), validation 2025-01..2025-06 (hyperparameters, thresholds),
test 2025-07..2026-08. Every month is predicted by models refit on all earlier target months (rolling origin).
Run from code/:  python benchmark.py  -> benchmark_results.json, figures in ../outputs/"""
import json, warnings, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.api as sm
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
warnings.filterwarnings("ignore")
SEED = 42; np.random.seed(SEED); OUT = "../outputs/"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": .6, "axes.axisbelow": True, "savefig.dpi": 200})
BLUE, ORANGE, GREY, DARK, GREEN, PURPLE = "#1f5a8a", "#c8641a", "#8a8a8a", "#222222", "#5b8e3e", "#7b4a9c"
raw = pd.read_csv("../data/processed/upi_remitter_top50_panel.csv")
months = sorted(raw.month.unique()); midx = {m: i for i, m in enumerate(months)}
d = raw[raw.is_bank].copy(); d["t"] = d.month.map(midx); d = d.sort_values(["entity", "t"]).reset_index(drop=True)
d["lnV"] = np.log(d.volume_mn); d["dlnV"] = d.groupby("entity").lnV.diff(); d.loc[d.groupby("entity").t.diff() != 1, "dlnV"] = np.nan
d["newpeak"] = ((d.lnV > d.groupby("entity").lnV.transform(lambda x: x.cummax().shift(1))) & d.groupby("entity").lnV.transform(lambda x: x.cummax().shift(1)).notna()).astype(int)
med = d.groupby("entity").td_pct.transform("median"); mad = d.groupby("entity").td_pct.transform(lambda x: (x - x.median()).abs().median()) * 1.4826
d["rz"] = (d.td_pct - med) / mad.replace(0, np.nan); d["spike"] = ((d.rz > 3) & (d.td_pct >= 1.0)).astype(int)
PUBLIC = ["State Bank of India", "Bank of Baroda", "Punjab National Bank", "Canara Bank", "Union Bank of India", "Bank of India", "Indian Bank", "Indian Overseas Bank", "Central Bank of India", "UCO Bank", "Bank of Maharashtra", "Punjab & Sind Bank"]
PRIVATE = ["HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank", "IndusInd Bank", "YES Bank", "IDFC FIRST Bank", "Federal Bank", "RBL Bank", "Bandhan Bank", "South Indian Bank", "Karur Vysya Bank", "City Union Bank", "Karnataka Bank", "Jammu & Kashmir Bank", "Tamilnad Mercantile Bank", "IDBI Bank"]
def grp(e):
    if e in PUBLIC: return "Public sector"
    if e in PRIVATE: return "Private sector"
    if "Small Finance" in e: return "Small finance"
    if "Payments Bank" in e: return "Payments bank"
    if "Gramin" in e or "Grameena" in e: return "Regional rural"
    return "Other"
d["group"] = d.entity.map(grp)
d["td_l1"] = d.groupby("entity").td_pct.shift(1); d["bd_l0"] = d.bd_pct
d["hist_mean"] = d.groupby("entity").td_pct.transform(lambda x: x.expanding().mean())
for c, s in [("next_td", "td_pct"), ("next_spike", "spike"), ("next_t", "t"), ("next_month", "month")]: d[c] = d.groupby("entity")[s].shift(-1)
M = d[(d.next_t == d.t + 1) & d.dlnV.notna() & d.td_l1.notna()].copy()
M["nm"] = M.next_month.str[5:].astype(int); M["sin_m"] = np.sin(2*np.pi*M.nm/12); M["cos_m"] = np.cos(2*np.pi*M.nm/12); M["peak_next"] = M.nm.isin([3, 10, 11]).astype(int)
for g in ["Public sector", "Private sector", "Small finance", "Payments bank", "Regional rural"]: M["g_" + g.split()[0]] = (M.group == g).astype(int)
FEATS = ["td_pct", "td_l1", "bd_l0", "lnV", "dlnV", "newpeak", "hist_mean", "sin_m", "cos_m", "peak_next"] + [c for c in M.columns if c.startswith("g_")]
M["next_spike"] = M.next_spike.astype(int)
VAL = [m for m in months if "2025-01" <= m <= "2025-06"]; TEST = [m for m in months if m >= "2025-07"]
sizes = dict(train=int((M.next_month < "2025-01").sum()), val=int(M.next_month.isin(VAL).sum()), test=int(M.next_month.isin(TEST).sum()),
             train_spikes=int(M[M.next_month < "2025-01"].next_spike.sum()), val_spikes=int(M[M.next_month.isin(VAL)].next_spike.sum()), test_spikes=int(M[M.next_month.isin(TEST)].next_spike.sum()))
print(sizes)
# ---------------- model factories (scaler fitted on training rows only)
def scaled(tr, te):
    sc = StandardScaler().fit(tr[FEATS]); return sc.transform(tr[FEATS]), sc.transform(te[FEATS])
def fit_predict_reg(kind, tr, te, hp):
    y = tr.next_td.clip(upper=10).values
    if kind == "glm":
        Xtr = sm.add_constant(tr[FEATS].astype(float)); Xte = sm.add_constant(te[FEATS].astype(float), has_constant="add")
        return sm.GLM((tr.next_td/100).clip(0, .5).values, Xtr, family=sm.families.Binomial()).fit().predict(Xte).values*100
    if kind == "xgb":
        m = xgb.XGBRegressor(n_estimators=hp["n"], max_depth=hp["depth"], learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=SEED, n_jobs=2, verbosity=0)
        return m.fit(tr[FEATS], y).predict(te[FEATS])
    if kind == "mlp":
        Xtr, Xte = scaled(tr, te); p = []
        for s in range(3):
            m = MLPRegressor(hidden_layer_sizes=hp["h"], alpha=hp["alpha"], learning_rate_init=.003, max_iter=400, early_stopping=True, validation_fraction=.15, n_iter_no_change=20, random_state=SEED+s)
            p.append(m.fit(Xtr, y).predict(Xte))
        return np.clip(np.mean(p, 0), 0, None)
def fit_predict_clf(kind, tr, te, hp):
    y = tr.next_spike.values
    if kind == "logit":
        Xtr, Xte = scaled(tr, te); return LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=2000).fit(Xtr, y).predict_proba(Xte)[:, 1]
    if kind == "xgb":
        pos = y.sum(); m = xgb.XGBClassifier(n_estimators=hp["n"], max_depth=hp["depth"], learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=SEED, scale_pos_weight=(len(y)-pos)/max(pos, 1), n_jobs=2, verbosity=0)
        return m.fit(tr[FEATS], y).predict_proba(te[FEATS])[:, 1]
    if kind == "mlp":
        Xtr, Xte = scaled(tr, te); pos_idx = np.where(y == 1)[0]; rng = np.random.RandomState(SEED)
        extra = rng.choice(pos_idx, size=min(len(y)//8, max(len(pos_idx)*8, 1)), replace=True)   # oversample spikes (train rows only)
        Xa = np.vstack([Xtr, Xtr[extra]]); ya = np.concatenate([y, y[extra]]); p = []
        for s in range(3):
            m = MLPClassifier(hidden_layer_sizes=hp["h"], alpha=hp["alpha"], learning_rate_init=.003, max_iter=400, early_stopping=True, validation_fraction=.15, n_iter_no_change=20, random_state=SEED+s)
            p.append(m.fit(Xa, ya).predict_proba(Xte)[:, 1])
        return np.mean(p, 0)
def rolling(fn, kind, hp, target_months):
    out = []
    for m in target_months:
        tr = M[M.next_month < m]; te = M[M.next_month == m]
        if te.empty: continue
        out.append(pd.Series(fn(kind, tr, te, hp), index=te.index))
    return pd.concat(out)
rmse = lambda y, p: float(np.sqrt(np.mean((y-p)**2)))
# ---------------- hyperparameter selection on validation window
grids = {("reg", "xgb"): [dict(depth=a, n=b) for a, b in itertools.product([2, 3, 4], [150, 300])],
         ("reg", "mlp"): [dict(h=a, alpha=b) for a, b in itertools.product([(16,), (32, 16)], [1e-3, 1e-1])],
         ("clf", "xgb"): [dict(depth=a, n=b) for a, b in itertools.product([2, 3], [150, 300])],
         ("clf", "mlp"): [dict(h=a, alpha=b) for a, b in itertools.product([(16,), (32, 16)], [1e-3, 1e-1])],
         ("clf", "logit"): [dict(C=c) for c in [.01, .1, 1.0]]}
V = M[M.next_month.isin(VAL)]; best = {}; tuning = []
for (task, kind), grid in grids.items():
    sc_ = []
    for hp in grid:
        p = rolling(fit_predict_reg if task == "reg" else fit_predict_clf, kind, hp, VAL)
        s = -rmse(V.loc[p.index].next_td.values, p.values) if task == "reg" else average_precision_score(V.loc[p.index].next_spike.values, p.values)
        sc_.append(s); tuning.append(dict(task=task, model=kind, hp=str(hp), score=float(abs(s) if task == "reg" else s)))
    best[(task, kind)] = grid[int(np.argmax(sc_))]
print(best)
# ---------------- test (rolling origin) + validation predictions for thresholds
T = M[M.next_month.isin(TEST)].copy(); y = T.next_td.values; ys = T.next_spike.values
P = {"Naive (last month)": T.td_pct.values, "Bank history mean": T.hist_mean.values}
P["Fractional logit"] = rolling(fit_predict_reg, "glm", {}, TEST).loc[T.index].values
P["XGBoost"] = rolling(fit_predict_reg, "xgb", best[("reg", "xgb")], TEST).loc[T.index].values
P["Neural network (MLP)"] = rolling(fit_predict_reg, "mlp", best[("reg", "mlp")], TEST).loc[T.index].values
# train (in-sample) accuracy: fit once on all rows with target month before the first test month
TR = M[M.next_month < TEST[0]]; yt = TR.next_td.values
PT = {"Naive (last month)": TR.td_pct.values, "Bank history mean": TR.hist_mean.values,
      "Fractional logit": fit_predict_reg("glm", TR, TR, {}), "XGBoost": fit_predict_reg("xgb", TR, TR, best[("reg", "xgb")]),
      "Neural network (MLP)": fit_predict_reg("mlp", TR, TR, best[("reg", "mlp")])}
perf = {}
for k in P:
    perf[k] = dict(train_rmse=rmse(yt, PT[k]), train_mae=float(np.mean(np.abs(yt-PT[k]))), test_rmse=rmse(y, P[k]), test_mae=float(np.mean(np.abs(y-P[k]))),
                   test_r2=float(1 - np.sum((y-P[k])**2)/np.sum((y-y.mean())**2)))
# paired bank-cluster bootstrap for RMSE difference vs naive
rng = np.random.RandomState(SEED); banks = T.entity.values; ub = np.unique(banks); idxs = {b: np.where(banks == b)[0] for b in ub}
boot = {k: [] for k in P}
for _ in range(1000):
    sel = np.concatenate([idxs[b] for b in rng.choice(ub, len(ub), replace=True)])
    for k in P: boot[k].append(rmse(y[sel], P[k][sel]) - rmse(y[sel], P["Naive (last month)"][sel]))
for k in P: perf[k]["diff_vs_naive_ci"] = [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))]
for k in ["Naive (last month)", "XGBoost", "Fractional logit", "Neural network (MLP)"]:
    e2 = (y-P[k])**2; perf[k]["top3_sq_share"] = float(np.sort(e2)[-3:].sum()/e2.sum()); perf[k]["rmse_ex_top3"] = float(np.sqrt(np.sort(e2)[:-3].mean()))
# classification: thresholds fixed on validation
Vc = M[M.next_month.isin(VAL)]; ysv = Vc.next_spike.values
def best_thr(p, yv):
    pr, rc, th = precision_recall_curve(yv, p); f1 = 2*pr[:-1]*rc[:-1]/np.maximum(pr[:-1]+rc[:-1], 1e-9); return float(th[int(np.argmax(f1))]) if len(th) else .5
C = {}
C["Naive (still in spike)"] = dict(score_test=T.spike.values.astype(float), thr=.5)
for name, kind in [("Logistic regression", "logit"), ("XGBoost classifier", "xgb"), ("Neural network (MLP)", "mlp")]:
    hp = best[("clf", kind)]
    pv = rolling(fit_predict_clf, kind, hp, VAL).loc[Vc.index].values
    pt = rolling(fit_predict_clf, kind, hp, TEST).loc[T.index].values
    Ptr = fit_predict_clf(kind, TR, TR, hp)
    C[name] = dict(score_test=pt, thr=best_thr(pv, ysv), train_auc=float(roc_auc_score(TR.next_spike.values, Ptr)), train_pr=float(average_precision_score(TR.next_spike.values, Ptr)))
cperf = {}
for k, v in C.items():
    s = v["score_test"]; pred = (s >= v["thr"]).astype(int); tn, fp, fn, tp = confusion_matrix(ys, pred, labels=[0, 1]).ravel()
    cperf[k] = dict(thr=v["thr"], tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1), accuracy=(tp+tn)/len(ys),
                    f1=2*tp/max(2*tp+fp+fn, 1), roc_auc=float(roc_auc_score(ys, s)), pr_auc=float(average_precision_score(ys, s)), flagged=int(pred.sum()),
                    train_auc=v.get("train_auc"), train_pr=v.get("train_pr"))
imp = None
m_ = xgb.XGBRegressor(n_estimators=best[("reg", "xgb")]["n"], max_depth=best[("reg", "xgb")]["depth"], learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=SEED, n_jobs=2, verbosity=0).fit(M[M.next_month < TEST[-1]][FEATS], M[M.next_month < TEST[-1]].next_td.clip(upper=10))
imp = pd.Series(m_.feature_importances_, index=FEATS).sort_values(ascending=False)
res = dict(sizes=sizes, best={f"{a}_{b}": {k: str(v) for k, v in hp.items()} for (a, b), hp in best.items()}, tuning=tuning, reg=perf, clf=cperf,
           importance=imp.round(3).to_dict(), n_test=int(len(T)), base_rate=float(ys.mean()), feats=FEATS, test_months=[TEST[0], TEST[-1]], val_months=[VAL[0], VAL[-1]])
json.dump(res, open("benchmark_results.json", "w"), indent=1, default=float)

# ---------------- illustrative watch list for the month after the data end (not validated; shows the deployed output)
L = d[(d.month == months[-1]) & d.dlnV.notna() & d.td_l1.notna()].copy(); L["nm"] = (int(months[-1][5:]) % 12) + 1
L["sin_m"] = np.sin(2*np.pi*L.nm/12); L["cos_m"] = np.cos(2*np.pi*L.nm/12); L["peak_next"] = int(L.nm.iloc[0] in (3, 10, 11))
for g in ["Public sector", "Private sector", "Small finance", "Payments bank", "Regional rural"]: L["g_" + g.split()[0]] = (L.group == g).astype(int)
Mtr = M[M.next_month <= months[-1]]
xr_ = xgb.XGBRegressor(n_estimators=best[("reg", "xgb")]["n"], max_depth=best[("reg", "xgb")]["depth"], learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=SEED, n_jobs=2, verbosity=0).fit(Mtr[FEATS], Mtr.next_td.clip(upper=10))
L["pred_td"] = xr_.predict(L[FEATS]); L["pred_spike_prob"] = fit_predict_clf("xgb", Mtr, L, best[("clf", "xgb")])
L["flag"] = (L.pred_spike_prob >= cperf["XGBoost classifier"]["thr"]).astype(int)
wl = L.sort_values("pred_td", ascending=False)[["entity", "group", "volume_mn", "td_pct", "pred_td", "pred_spike_prob", "flag"]]
wl.to_csv("../outputs/watchlist_next_month.csv", index=False); res["watchlist"] = dict(month=months[-1], n=int(len(wl)), n_flag=int(wl.flag.sum()), top=wl.head(8).round(3).to_dict(orient="records"),
      top10_pred_vol_share=float(wl.head(10).volume_mn.sum()/wl.volume_mn.sum()))
json.dump(res, open("benchmark_results.json", "w"), indent=1, default=float)
print(wl.head(10).round(3).to_string())
# ---------------- figures
def save(fig, n): fig.tight_layout(); fig.savefig(OUT + n, bbox_inches="tight", facecolor="white"); plt.close(fig)
names = list(P); cols = [GREY, GREY, BLUE, ORANGE, GREEN]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for a, met, lab in [(ax[0], "rmse", "RMSE (pp)"), (ax[1], "mae", "MAE (pp)")]:
    x = np.arange(len(names)); w = .38
    a.bar(x-w/2, [perf[k]["train_"+met] for k in names], w, color="#c9d6e3", label="Training", edgecolor=DARK, lw=.4)
    a.bar(x+w/2, [perf[k]["test_"+met] for k in names], w, color=cols, label="Test", edgecolor=DARK, lw=.4)
    a.set_xticks(x); a.set_xticklabels([n.replace(" (", "\n(").replace("Bank history", "Bank\nhistory").replace("Fractional", "Fractional\n") for n in names], fontsize=6.8); a.set_ylabel(lab)
    for i, k in enumerate(names): a.text(i+w/2, perf[k]["test_"+met]+.01, f"{perf[k]['test_'+met]:.2f}", ha="center", fontsize=7)
ax[0].legend(fontsize=7.5, frameon=False)
save(fig, "fig11_model_compare.png")
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0)); cc = {"Naive (still in spike)": GREY, "Logistic regression": BLUE, "XGBoost classifier": ORANGE, "Neural network (MLP)": GREEN}
for k, v in C.items():
    s = v["score_test"]; fpr, tpr, _ = roc_curve(ys, s); pr, rc, _ = precision_recall_curve(ys, s)
    ax[0].plot(fpr, tpr, color=cc[k], lw=1.6, label=f"{k} (AUC {cperf[k]['roc_auc']:.2f})"); ax[1].plot(rc, pr, color=cc[k], lw=1.6, label=f"{k} (AP {cperf[k]['pr_auc']:.2f})")
ax[0].plot([0, 1], [0, 1], color=DARK, ls=":", lw=1); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(fontsize=6.3, frameon=False, loc="lower right")
ax[1].axhline(ys.mean(), color=DARK, ls=":", lw=1); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend(fontsize=6.3, frameon=False, loc="upper right"); ax[1].set_ylim(0, 1.02)
ax[0].set_title("ROC curve", fontsize=9, loc="left"); ax[1].set_title("Precision-recall curve (dotted = base rate)", fontsize=9, loc="left")
save(fig, "fig12_roc_pr.png")
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.2))
for a, (k, v) in zip(ax, cperf.items()):
    cmx = np.array([[v["tn"], v["fp"]], [v["fn"], v["tp"]]]); a.imshow(cmx, cmap="Blues", vmax=max(cmx.max(), 1)*1.3); a.grid(False)
    for i in range(2):
        for j in range(2): a.text(j, i, cmx[i, j], ha="center", va="center", fontsize=10, color="white" if cmx[i, j] > cmx.max()*.5 else DARK)
    a.set_xticks([0, 1]); a.set_xticklabels(["No spike", "Spike"], fontsize=7); a.set_yticks([0, 1]); a.set_yticklabels(["No spike", "Spike"], fontsize=7)
    a.set_title(k.replace(" (", "\n("), fontsize=7.5); a.set_xlabel("Predicted", fontsize=7.5)
ax[0].set_ylabel("Actual", fontsize=7.5)
save(fig, "fig13_confusion.png")
bestk = min(["Fractional logit", "XGBoost", "Neural network (MLP)"], key=lambda k: perf[k]["test_rmse"])
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.7)); p = P[bestk]
ax[0].scatter(y, p, s=8, color=BLUE, alpha=.45, lw=0); mx = max(y.max(), p.max()); ax[0].plot([0, mx], [0, mx], color=DARK, ls=":", lw=1)
ax[0].set_xlabel("Actual next-month TD (%)"); ax[0].set_ylabel(f"Predicted ({bestk})"); ax[0].set_title("Predicted vs. actual", fontsize=9, loc="left")
ax[1].hist(y-p, bins=40, color=BLUE, alpha=.9); ax[1].set_yscale("log"); ax[1].set_xlabel("Residual (actual - predicted), pp"); ax[1].set_ylabel("Bank-months (log)"); ax[1].set_title("Residuals", fontsize=9, loc="left")
LAB = {"g_Regional": "Regional rural group", "td_pct": "TD this month", "hist_mean": "Bank history mean", "td_l1": "TD last month", "newpeak": "New volume peak", "lnV": "Log volume", "bd_l0": "BD this month", "cos_m": "Calendar (cos)", "sin_m": "Calendar (sin)", "dlnV": "Volume growth", "peak_next": "Peak month", "g_Payments": "Payments bank group", "g_Public": "Public sector group", "g_Private": "Private sector group", "g_Small": "Small finance group"}
top = imp.head(8)[::-1]; ax[2].barh([LAB.get(i, i) for i in top.index], top.values, color=ORANGE); ax[2].tick_params(axis="y", labelsize=7); ax[2].set_xlabel("XGBoost importance"); ax[2].set_title("Feature importance", fontsize=9, loc="left")
save(fig, "fig14_diagnostics.png")
print(json.dumps({k: {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()} for k, v in perf.items()}, indent=0))
print(json.dumps({k: {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()} for k, v in cperf.items()}, indent=0)); print(bestk)
