"""Export standalone accuracy/cost figures from the completed experiment only."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from repar.io import read_json

p = argparse.ArgumentParser()
p.add_argument("--root", default="artifacts/experiment")
args = p.parse_args()
root = Path(args.root)
s = read_json(root / "summary.json")
methods = ["direct", "bm25", "repar_tree", "repar"]
names = ["Direct context", "BM25 RAG", "RePAR (graph disabled)", "RePAR"]
colors = ["#75808b", "#57a2bc", "#d3aa52", "#386b4f"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
values = [s["methods"][m]["overall"]["accuracy"] for m in methods]
intervals = np.array([s["methods"][m]["overall"]["wilson_95"] for m in methods])
axes[0].barh(names, values, color=colors)
axes[0].errorbar(values, names, xerr=np.array([np.array(values)-intervals[:,0], intervals[:,1]-np.array(values)]),
                 fmt="none", ecolor="#25333a", capsize=4, linewidth=1)
axes[0].axvline(44, linestyle="--", color="#8e4960", label="GPT-4o: 44% (paper, historical)")
axes[0].set_xlim(0, max(65, intervals[:,1].max()+8))
axes[0].set_xlabel("Accuracy (%) with Wilson 95% interval")
axes[0].set_title("LongBench v2 Multi-Doc QA (all 125 questions)", fontsize=10)
axes[0].legend(fontsize=7, loc="lower right")
for i, value in enumerate(values):
    axes[0].text(value-1, i, f"{value:.1f}%", ha="right", va="center", color="white", fontsize=9)
for m, name, color in zip(methods, names, colors):
    r = s["methods"][m]["overall"]
    x, y = r["mean_prompt_tokens"]/1000, r["accuracy"]
    axes[1].scatter(x, y, s=90, color=color, edgecolor="white", zorder=3)
    axes[1].annotate(name, (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
axes[1].axhline(44, linestyle="--", color="#8e4960", linewidth=1)
axes[1].set_xlabel("Mean input tokens per question, all calls (thousands)")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(max(0, min(values)-10), min(100, max(values)+15))
axes[1].set_title("Measured accuracy and inference input cost", fontsize=10)
axes[1].grid(alpha=.2)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "results.png", dpi=220)
fig.savefig(root / "results.svg")
print("Saved results.png and results.svg")
