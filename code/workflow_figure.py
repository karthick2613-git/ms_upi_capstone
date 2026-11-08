"""Architecture / workflow diagram (Figure 1). Run from code/: python workflow_figure.py"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"]})
stages = [("1  Data", "NPCI Top-50 Member\nPerformance tables\n(33 Excel files, remitter side)\nSQLite database", "#dbe7f3", True),
          ("2  Preprocessing", "Name harmonizing\nUnit conversion (0–100)\nDuplicate removal\nScaling and encoding\nMissing-value rules", "#e6e6e6", True),
          ("3  EDA", "Distributions, trends\nBank groups, seasonality\nCorrelation heatmap\nSpike detection", "#fbe3cf", True),
          ("4  Feature\nEngineering", "Log volume, growth\nNew-peak flag, lags\nBank history mean\nCalendar sin/cos, Post", "#e6e6e6", True),
          ("5  Model", "RQ1–RQ4: fractional logit,\nITS, mixed model, Cox\nBenchmark: baselines,\nXGBoost, neural network", "#d9ead0", True),
          ("6  Evaluation", "Rolling-origin test\nRMSE, MAE, ROC, PR\nConfusion matrix\nBootstrap intervals", "#d9ead0", True),
          ("7  Deployment", "Monthly refresh job\nDashboard + risk list\nNPCI / bank users\n(proposed design)", "#e5d9ee", False)]
fig, ax = plt.subplots(figsize=(7.4, 3.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
w, h = 21, 17; ytop, ybot = 31, 4
pos = [(2, ytop), (27, ytop), (52, ytop), (77, ytop), (2, ybot), (27, ybot), (52, ybot)]
for (t, body, col, impl), (x, y) in zip(stages, pos):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=col, ec="#222222", lw=1.3, ls="-" if impl else (0, (4, 2))))
    ax.text(x+w/2, y+h-2.6, t.replace("\n", " "), ha="center", va="center", fontsize=7.6, fontweight="bold", color="#111111")
    ax.text(x+w/2, y+6.4, body, ha="center", va="center", fontsize=6.0, color="#222222", linespacing=1.15)
def arrow(p, q): ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color="#333333", lw=1.2))
def line(p, q): ax.plot([p[0], q[0]], [p[1], q[1]], color="#333333", lw=1.2)
for i in range(3): arrow((pos[i][0]+w+.4, ytop+h/2), (pos[i+1][0]-.4, ytop+h/2))
line((87.5, ytop-.4), (87.5, 27.5)); line((87.5, 27.5), (12.5, 27.5)); arrow((12.5, 27.5), (12.5, ybot+h+.5))
for i in range(4, 6): arrow((pos[i][0]+w+.4, ybot+h/2), (pos[i+1][0]-.4, ybot+h/2))
ax.text(50, 48.8, "Stages 1–4 are shared by RQ1–RQ4 and the benchmark; stage 5 splits by question. A new monthly NPCI table re-enters at stage 1.", fontsize=6.2, ha="center", color="#333333", style="italic")
leg = [Patch(fc="#dbe7f3", ec="#222222", label="Data"), Patch(fc="#e6e6e6", ec="#222222", label="Processing"), Patch(fc="#fbe3cf", ec="#222222", label="Analysis"),
       Patch(fc="#d9ead0", ec="#222222", label="Modeling and evaluation"), Patch(fc="#e5d9ee", ec="#222222", ls="--", label="Proposed, not yet live")]
ax.legend(handles=leg, loc="center", bbox_to_anchor=(0.885, 0.21), fontsize=6.2, frameon=True, edgecolor="#999999", title="Legend", title_fontsize=6.5, borderpad=.6)
fig.savefig("../outputs/fig_workflow.png", dpi=220, bbox_inches="tight", facecolor="white")
