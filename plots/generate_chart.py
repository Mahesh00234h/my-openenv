"""
Generate two charts:
  1. plots/difficulty_breakdown.png  — random agent across all 3 difficulties
  2. plots/full_comparison.png       — random vs GPT-4o-mini vs trained (medium_cos)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT   = Path(__file__).parent.parent
PLOTS  = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

# ── Load data ────────────────────────────────────────────────────────────────

smoke   = load_json(ROOT / "results" / "smoke_test_results.json")
llm_raw = load_json(ROOT / "results" / "baseline_llm.json")

difficulties = ["easy_cos", "medium_cos", "hard_cos"]
labels_diff  = ["Easy", "Medium", "Hard"]

# ════════════════════════════════════════════════════════════════════════════
# CHART 1 — difficulty_breakdown.png
# ════════════════════════════════════════════════════════════════════════════

email_vals  = [smoke.get(d, {}).get("email",      0) for d in difficulties]
cal_vals    = [smoke.get(d, {}).get("calendar",   0) for d in difficulties]
deleg_vals  = [smoke.get(d, {}).get("delegation", 0) for d in difficulties]
combined_c1 = [
    round(0.40 * smoke.get(d, {}).get("email", 0)
        + 0.35 * smoke.get(d, {}).get("calendar", 0)
        + 0.25 * smoke.get(d, {}).get("delegation", 0), 3)
    for d in difficulties
]

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
fig1.patch.set_facecolor("#f9f9f9")

x     = np.arange(len(labels_diff))
width = 0.25

for ax in (ax1, ax2):
    ax.set_facecolor("#f9f9f9")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)

b_email = ax1.bar(x - width, email_vals,  width, label="Email",      color="#4C72B0", alpha=0.88, edgecolor="white")
b_cal   = ax1.bar(x,         cal_vals,    width, label="Calendar",   color="#55A868", alpha=0.88, edgecolor="white")
b_deleg = ax1.bar(x + width, deleg_vals,  width, label="Delegation", color="#C44E52", alpha=0.88, edgecolor="white")

for bars in (b_email, b_cal, b_deleg):
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")

ax1.axhline(y=0.18, color="#999999", linestyle="--", linewidth=1.2)
ax1.text(len(labels_diff) - 0.45, 0.19, "Random Floor", color="#999999", fontsize=8)
ax1.set_ylim(0.0, 1.15)
ax1.set_ylabel("Mean Reward per Decision", fontsize=11, labelpad=8)
ax1.set_title("AI Chief of Staff — Random Agent Baseline by Difficulty",
              fontsize=11, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels_diff, fontsize=11)
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax1.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="#cccccc")

diff_colors = ["#27ae60", "#e67e22", "#e74c3c"]
bars_c = ax2.bar(x, combined_c1, width=0.45, color=diff_colors, alpha=0.88, edgecolor="white")
for bar, val in zip(bars_c, combined_c1):
    ax2.annotate(f"{val:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, val),
        xytext=(0, 4), textcoords="offset points",
        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333333")

ax2.set_ylim(0.0, 1.15)
ax2.set_ylabel("Combined Reward", fontsize=11, labelpad=8)
ax2.set_title("Combined Reward by Difficulty", fontsize=11, fontweight="bold", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(labels_diff, fontsize=11)
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
legend_els = [mpatches.Patch(facecolor=c, label=l, alpha=0.88)
              for c, l in zip(diff_colors, labels_diff)]
ax2.legend(handles=legend_els, fontsize=9, loc="upper right",
           framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout(pad=2.0)
out1 = PLOTS / "difficulty_breakdown.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved to plots/difficulty_breakdown.png")

# ════════════════════════════════════════════════════════════════════════════
# CHART 2 — full_comparison.png  (medium_cos, 3 agents)
# ════════════════════════════════════════════════════════════════════════════

TASK = "medium_cos"

# Random scores from smoke test
rand = smoke.get(TASK, {})
rand_scores = {
    "email":      rand.get("email",      0.0),
    "calendar":   rand.get("calendar",   0.0),
    "delegation": rand.get("delegation", 0.0),
    "combined":   round(0.40 * rand.get("email", 0)
                      + 0.35 * rand.get("calendar", 0)
                      + 0.25 * rand.get("delegation", 0), 4),
}

# LLM scores — fall back to 0 if not yet run
llm_task = llm_raw.get("results", {}).get(TASK, {})
llm_has_data = bool(llm_task)
llm_scores = {
    "email":      llm_task.get("email",      0.0),
    "calendar":   llm_task.get("calendar",   0.0),
    "delegation": llm_task.get("delegation", 0.0),
    "combined":   llm_task.get("combined",   0.0),
}

# Trained — placeholder
trained_scores = {"email": 0.0, "calendar": 0.0, "delegation": 0.0, "combined": 0.0}

categories = ["Email", "Calendar", "Delegation", "Combined"]
keys       = ["email", "calendar", "delegation", "combined"]

rand_vals    = [rand_scores[k]    for k in keys]
llm_vals     = [llm_scores[k]     for k in keys]
trained_vals = [trained_scores[k] for k in keys]

x2    = np.arange(len(categories))
w2    = 0.25

fig2, ax = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor("#f9f9f9")
ax.set_facecolor("#f9f9f9")
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
ax.set_axisbelow(True)

b_rand    = ax.bar(x2 - w2, rand_vals,    w2, label="Random Agent",              color="#999999", alpha=0.88, edgecolor="white")
b_llm     = ax.bar(x2,      llm_vals,     w2, label="GPT-4o-mini (before training)", color="#FF8C00", alpha=0.88, edgecolor="white")
b_trained = ax.bar(x2 + w2, trained_vals, w2, label="After Training (TBD)",      color="#2E8B57", alpha=0.88, edgecolor="white")

# Value labels
def label_bars(bars, is_tbd=False):
    for bar in bars:
        h = bar.get_height()
        cx = bar.get_x() + bar.get_width() / 2
        if is_tbd:
            ax.annotate("TBD",
                xy=(cx, 0.01), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color="#555555", style="italic")
        else:
            lbl = f"{h:.2f}" if h > 0.005 else "—"
            ax.annotate(lbl,
                xy=(cx, max(h, 0.005)), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")

label_bars(b_rand)
label_bars(b_llm,  is_tbd=not llm_has_data)
label_bars(b_trained, is_tbd=True)

# Dashed random baseline lines per group
for i, rv in enumerate(rand_vals):
    ax.plot([x2[i] - w2 * 1.8, x2[i] + w2 * 1.8], [rv, rv],
            color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)

# "Run GRPO" annotation on trained bars group
ax.annotate("← Run GRPO to fill these in",
    xy=(x2[-1] + w2, 0.05),
    xytext=(x2[-1] + w2 + 0.15, 0.18),
    fontsize=8.5, color="#2E8B57",
    arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.2))

ax.set_ylim(0.0, 1.20)
ax.set_ylabel("Mean Reward per Decision", fontsize=12, labelpad=10)
ax.set_title("AI Chief of Staff — Agent Performance Comparison  (medium_cos)",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xticks(x2)
ax.set_xticklabels(categories, fontsize=12)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.legend(fontsize=10, loc="upper left", framealpha=0.9, edgecolor="#cccccc")

if not llm_has_data:
    fig2.text(0.5, 0.01,
        "GPT-4o-mini bars will populate after running: python3 inference.py",
        ha="center", fontsize=8.5, color="#FF8C00")

plt.tight_layout(rect=[0, 0.04, 1, 1])
out2 = PLOTS / "full_comparison.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved to plots/full_comparison.png")
