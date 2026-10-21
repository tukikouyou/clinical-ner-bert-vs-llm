# -*- coding: utf-8 -*-
"""論文図: paper/figs/*.pdf|png  (主条件 = 生成予算 4096 "4k"; LAU 設定 = 900)"""
import os, json, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); STATS = os.path.join(HERE, "../stats"); FIG = os.path.join(HERE, "../paper/figs")
os.makedirs(FIG, exist_ok=True)
def S(n):
    f = os.path.join(STATS, n + ".json"); return json.load(open(f)) if os.path.exists(f) else None
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "pdf.fonttype": 42})
mm = 1 / 25.4
# (label, zs-prefix, ft stem) ; seed runs: <stem>-ft, <stem>-s1-ft, <stem>-s2-ft (+ "-4k")
MODELS = [("Med42-70B*", "med42-70b", "med42-70b"), ("Llama-3.3-70B", "llama-3.3-70b", "llama-3.3-70b"),
          ("MedGemma-27B*", "medgemma-27b", "medgemma-27b"), ("Qwen3-30B-A3B", "qwen", "qwen"),
          ("LLM-jp-4-32B", "llm-jp-4-32b", "llm-jp-4-32b"), ("SIP-jmed-13B*", "sip-jmed-13b", "sip-jmed-13b"),
          ("LLM-jp-1.8B", "llm-jp-1.8b", "llm-jp-1.8b")]
def seeds(stem, budget):
    names = [f"{stem}-ft{budget}", f"{stem}-s1-ft{budget}", f"{stem}-s2-ft{budget}"]
    if stem == "gemma-3-27b" and budget == "":
        names[0] = "gemma-3-27b-s42-ft"
    return [S(n) for n in names]
def mean_sd(vals):
    v = [x for x in vals if x is not None]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else (0.0, 0.0)
def zs_v2(p):
    return S(p + "-zs@v2") or S(p + "-zs")

# ---- Fig 1: ZS (parser v2) / ZS constrained / FT 900 / FT 4k (mean±SD over seeds), relaxed and strict
fig, axes = plt.subplots(1, 2, figsize=(170 * mm, 64 * mm), sharey=True)
for ax, mode in zip(axes, ["relaxed", "strict"]):
    x = np.arange(len(MODELS)); w = 0.2
    zs = [zs_v2(p)[mode]["F1"] for _, p, _ in MODELS]
    cg = [S(p + "-zs-guided")[mode]["F1"] for _, p, _ in MODELS]
    f9 = [mean_sd([s[mode]["F1"] if s else None for s in seeds(stem, "")]) for _, _, stem in MODELS]
    f4 = [mean_sd([s[mode]["F1"] if s else None for s in seeds(stem, "-4k")]) for _, _, stem in MODELS]
    ax.bar(x - 1.5 * w, zs, w, color="#c6dbef", label="Zero-shot, prompt only")
    ax.bar(x - 0.5 * w, cg, w, color="#6baed6", label="Zero-shot + constrained decoding")
    ax.bar(x + 0.5 * w, [m for m, _ in f9], w, yerr=[d for _, d in f9], capsize=1.5, color="#fdae6b", label="Fine-tuned, 900-token budget")
    ax.bar(x + 1.5 * w, [m for m, _ in f4], w, yerr=[d for _, d in f4], capsize=1.5, color="#e6550d", label="Fine-tuned, 4,096-token budget")
    ax.axhline(S("BERT-UTH")[mode]["F1"], color="0.2", ls=(0, (5, 3)), lw=1, label="UTH-BERT (fine-tuned)")
    ax.set_xticks(x); ax.set_xticklabels([m for m, _, _ in MODELS], rotation=35, ha="right")
    ax.set_title(f"{mode.capitalize()} match"); ax.set_ylim(0, 1.0); ax.grid(axis="y", lw=0.4, alpha=0.5)
axes[0].set_ylabel("F1 (micro)"); h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=7)
fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(os.path.join(FIG, "fig_overview.pdf")); fig.savefig(os.path.join(FIG, "fig_overview.png"), dpi=300); plt.close(fig)

# ---- Fig 2: base-aligned pairs per seed (main budget 4k)
PAIRS = [("13B pair", "sip-jmed-13b", "llmjp-3.1-13b"), ("27B pair", "medgemma-27b", "gemma-3-27b"), ("70B pair", "med42-70b", "llama-3.3-70b")]
fig, axes = plt.subplots(1, 2, figsize=(170 * mm, 58 * mm))
for ax, mode in zip(axes, ["relaxed", "strict"]):
    for i, (name, med, base) in enumerate(PAIRS):
        m = [s[mode]["F1"] if s else None for s in seeds(med, "-4k")]; b = [s[mode]["F1"] if s else None for s in seeds(base, "-4k")]
        for sm, sb in zip(m, b):
            if sm is None or sb is None:
                continue
            ax.scatter([i - 0.12], [sm], color="#e6550d", zorder=3, s=22); ax.scatter([i + 0.12], [sb], color="#3182bd", zorder=3, s=22, marker="s")
            ax.plot([i - 0.12, i + 0.12], [sm, sb], color="0.6", lw=0.6, zorder=2)
    ax.scatter([], [], color="#e6550d", s=22, label="medical"); ax.scatter([], [], color="#3182bd", s=22, marker="s", label="general-purpose base")
    ax.set_xticks(range(3)); ax.set_xticklabels([n for n, _, _ in PAIRS], fontsize=8); ax.set_title(f"{mode.capitalize()} match"); ax.grid(axis="y", lw=0.4, alpha=0.5)
axes[0].set_ylabel("F1 (micro), seeds 42/1/2"); axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_pairs.pdf")); fig.savefig(os.path.join(FIG, "fig_pairs.png"), dpi=300); plt.close(fig)

# ---- Fig 3: label frequency vs F1 (Med42-70B-ft, seed 42, 4k)
e = json.load(open(os.path.join(STATS, "extra.json")))
per = e.get("per_label_med42_4k", e["per_label_med42"])
fig, ax = plt.subplots(figsize=(120 * mm, 70 * mm))
for t, v in per.items():
    oth = t.endswith("-others")
    ax.scatter(v["n"], v["relaxed"][2], s=14, color="#d62728" if oth else "#1f77b4", alpha=0.8, zorder=3)
ax.scatter([], [], color="#1f77b4", s=14, label="other labels"); ax.scatter([], [], color="#d62728", s=14, label="-others (non-patient subject)")
ax.set_xscale("log"); ax.set_xlabel("Gold entities in test set (log scale)"); ax.set_ylabel("Relaxed F1 (fine-tuned Med42-70B)")
ax.grid(lw=0.4, alpha=0.5); ax.legend(frameon=False, loc="lower right")
for t in ["state-patient", "value-others", "state-others", "PN-Negative-patient", "clinical_test", "device", "time_span"]:
    v = per[t]; ax.annotate(t, (v["n"], v["relaxed"][2]), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color="0.25")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_scatter.pdf")); fig.savefig(os.path.join(FIG, "fig_scatter.png"), dpi=300); plt.close(fig)

# ---- Fig 4: truncation at 900: recall on truncated vs non-truncated sentences
tr = e["truncation"]
names = [m for m, _, _ in MODELS] + ["UTH-BERT"]
keys = ["Med42-70B", "Llama-3.3-70B", "MedGemma-27B", "Qwen3-30B-A3B", "LLM-jp-4-32B", "SIP-jmed-13B", "LLM-jp-1.8B", "UTH-BERT (Med42 の截断文で分割)"]
fig, ax = plt.subplots(figsize=(120 * mm, 58 * mm))
x = np.arange(len(keys)); w = 0.38
ax.bar(x - w / 2, [tr[k]["rest"]["R"] for k in keys], w, color="#74c476", label="sentences completed within 900 tokens (~80%)")
ax.bar(x + w / 2, [tr[k]["trunc"]["R"] for k in keys], w, color="#fd8d3c", label="sentences truncated at 900 tokens (~20%, 52% of gold entities)")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right"); ax.set_ylabel("Recall (relaxed)"); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.grid(axis="y", lw=0.4, alpha=0.5)
ax.legend(frameon=False, loc="upper left", fontsize=6.8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_trunc.pdf")); fig.savefig(os.path.join(FIG, "fig_trunc.png"), dpi=300); plt.close(fig)
print("figs written")
