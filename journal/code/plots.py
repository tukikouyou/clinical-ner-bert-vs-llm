# -*- coding: utf-8 -*-
"""論文図: paper/figs/*.pdf|png"""
import os, json, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); STATS = os.path.join(HERE, "../stats"); FIG = os.path.join(HERE, "../paper/figs")
os.makedirs(FIG, exist_ok=True)
S = lambda n: json.load(open(os.path.join(STATS, n + ".json")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "pdf.fonttype": 42})
mm = 1 / 25.4
MODELS = [("Med42-70B*", "med42-70b", ["med42-70b-ft", "med42-70b-s1-ft", "med42-70b-s2-ft"]),
          ("Llama-3.3-70B", "llama-3.3-70b", ["llama-3.3-70b-ft", "llama-3.3-70b-s1-ft", "llama-3.3-70b-s2-ft"]),
          ("MedGemma-27B*", "medgemma-27b", ["medgemma-27b-ft", "medgemma-27b-s1-ft", "medgemma-27b-s2-ft"]),
          ("Qwen3-30B-A3B", "qwen", ["qwen-ft", "qwen-s1-ft", "qwen-s2-ft"]),
          ("LLM-jp-4-32B", "llm-jp-4-32b", ["llm-jp-4-32b-ft", "llm-jp-4-32b-s1-ft", "llm-jp-4-32b-s2-ft"]),
          ("SIP-jmed-13B*", "sip-jmed-13b", ["sip-jmed-13b-ft", "sip-jmed-13b-s1-ft", "sip-jmed-13b-s2-ft"]),
          ("LLM-jp-1.8B", "llm-jp-1.8b", ["llm-jp-1.8b-ft", "llm-jp-1.8b-s1-ft", "llm-jp-1.8b-s2-ft"])]
# ---- Fig 1: ZS (parser v2) / ZS constrained / FT mean±SD, relaxed and strict
def zs_v2(p):
    n = p + "-zs@v2"
    return S(n) if os.path.exists(os.path.join(STATS, n + ".json")) else S(p + "-zs")
fig, axes = plt.subplots(1, 2, figsize=(170 * mm, 62 * mm), sharey=True)
for ax, mode, bert in zip(axes, ["relaxed", "strict"], [S("BERT-UTH"), S("BERT-UTH")]):
    x = np.arange(len(MODELS)); w = 0.27
    zs = [zs_v2(p)[mode]["F1"] for _, p, _ in MODELS]
    cg = [S(p + "-zs-guided")[mode]["F1"] for _, p, _ in MODELS]
    ft = [[S(r)[mode]["F1"] for r in runs] for _, _, runs in MODELS]
    ax.bar(x - w, zs, w, color="#9ecae1", label="Zero-shot (prompt only)")
    ax.bar(x, cg, w, color="#3182bd", label="Zero-shot + constrained decoding")
    ax.bar(x + w, [st.mean(v) for v in ft], w, yerr=[st.stdev(v) for v in ft], capsize=2, color="#e6550d", label="QLoRA fine-tuned (mean of 3 seeds)")
    ax.axhline(bert[mode]["F1"], color="0.2", ls=(0, (5, 3)), lw=1, label=f"UTH-BERT (fine-tuned)")
    ax.set_xticks(x); ax.set_xticklabels([m for m, _, _ in MODELS], rotation=35, ha="right")
    ax.set_title(f"{mode.capitalize()} match"); ax.set_ylim(0, 0.95); ax.grid(axis="y", lw=0.4, alpha=0.5)
axes[0].set_ylabel("F1 (micro)"); h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(FIG, "fig_overview.pdf")); fig.savefig(os.path.join(FIG, "fig_overview.png"), dpi=300); plt.close(fig)
# ---- Fig 2: base-aligned pairs per seed
PAIRS = [("13B: SIP-jmed vs LLM-jp-3.1", ["sip-jmed-13b-ft", "sip-jmed-13b-s1-ft", "sip-jmed-13b-s2-ft"], ["llmjp-3.1-13b-ft", "llmjp-3.1-13b-s1-ft", "llmjp-3.1-13b-s2-ft"]),
         ("27B: MedGemma vs Gemma-3", ["medgemma-27b-ft", "medgemma-27b-s1-ft", "medgemma-27b-s2-ft"], ["gemma-3-27b-s42-ft", "gemma-3-27b-s1-ft", "gemma-3-27b-s2-ft"]),
         ("70B: Med42 vs Llama-3.3", ["med42-70b-ft", "med42-70b-s1-ft", "med42-70b-s2-ft"], ["llama-3.3-70b-ft", "llama-3.3-70b-s1-ft", "llama-3.3-70b-s2-ft"])]
fig, axes = plt.subplots(1, 2, figsize=(170 * mm, 58 * mm))
for ax, mode in zip(axes, ["relaxed", "strict"]):
    for i, (name, med, base) in enumerate(PAIRS):
        m = [S(r)[mode]["F1"] for r in med]; b = [S(r)[mode]["F1"] for r in base]
        ax.scatter([i - 0.12] * 3, m, color="#e6550d", zorder=3, s=22, label="medical" if i == 0 else None)
        ax.scatter([i + 0.12] * 3, b, color="#3182bd", zorder=3, s=22, marker="s", label="general base" if i == 0 else None)
        for sm, sb in zip(m, b):
            ax.plot([i - 0.12, i + 0.12], [sm, sb], color="0.6", lw=0.6, zorder=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["13B pair", "27B pair", "70B pair"], fontsize=8); ax.set_title(f"{mode.capitalize()} match"); ax.grid(axis="y", lw=0.4, alpha=0.5)
axes[0].set_ylabel("F1 (micro), seeds 42/1/2"); axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_pairs.pdf")); fig.savefig(os.path.join(FIG, "fig_pairs.png"), dpi=300); plt.close(fig)
# ---- Fig 3: label frequency vs F1 (Med42-70B-ft, seed 42)
e = json.load(open(os.path.join(STATS, "extra.json")))
per = e["per_label_med42"]
fig, ax = plt.subplots(figsize=(120 * mm, 70 * mm))
for t, v in per.items():
    oth = t.endswith("-others")
    ax.scatter(v["n"], v["relaxed"][2], s=14, color="#d62728" if oth else "#1f77b4", alpha=0.8, zorder=3)
ax.scatter([], [], color="#1f77b4", s=14, label="other labels"); ax.scatter([], [], color="#d62728", s=14, label="-others (non-patient subject)")
ax.set_xscale("log"); ax.set_xlabel("Gold entities in test set (log scale)"); ax.set_ylabel("Relaxed F1 (Med42-70B, fine-tuned)")
ax.grid(lw=0.4, alpha=0.5); ax.legend(frameon=False, loc="lower right")
for t in ["state-patient", "value-others", "state-others", "PN-Negative-patient", "clinical_test", "device", "time_span"]:
    v = per[t]; ax.annotate(t, (v["n"], v["relaxed"][2]), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color="0.25")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_scatter.pdf")); fig.savefig(os.path.join(FIG, "fig_scatter.png"), dpi=300); plt.close(fig)
# ---- Fig 4: truncation: recall on truncated vs non-truncated sentences
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
