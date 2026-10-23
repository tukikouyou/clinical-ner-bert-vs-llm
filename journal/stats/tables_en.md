#### Table A. Fine-tuned models, main condition (generation budget 4,096 tokens; 2,600 for the LLM-jp-3.1 family), seed 42

95% CI: document-level bootstrap (36 documents, 2,000 resamples). min2/min3: relaxed F1 with a minimum substring length of 2/3 characters. Macro: mean F1 over the 48 labels with ≥ 20 test entities. Merged: relaxed F1 after merging the `-patient`/`-others` suffixes. Exact/TP: share of relaxed true positives whose surface is an exact match. Trunc.: sentences cut off at the generation budget.

| Model | Type | Budget | Relaxed F1 | 95% CI | P | R | Strict F1 | 95% CI | min2 | min3 | Macro48 | Macro98 | Merged | Exact/TP | Trunc./433 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | medical | 4096 | 0.905 | [0.891, 0.919] | 0.912 | 0.899 | 0.869 | [0.851, 0.885] | 0.899 | 0.885 | 0.848 | 0.658 | 0.920 | 0.954 | 2 |
| Llama-3.3-70B | general | 4096 | 0.909 | [0.898, 0.919] | 0.915 | 0.902 | 0.871 | [0.856, 0.884] | 0.903 | 0.890 | 0.862 | 0.675 | 0.920 | 0.954 | 2 |
| MedGemma-27B | medical | 4096 | 0.899 | [0.885, 0.913] | 0.906 | 0.893 | 0.859 | [0.842, 0.875] | 0.893 | 0.876 | 0.844 | 0.655 | 0.914 | 0.951 | 2 |
| Gemma-3-27B (base of MedGemma) | general | 4096 | 0.909 | [0.897, 0.921] | 0.914 | 0.905 | 0.873 | [0.858, 0.888] | 0.903 | 0.889 | 0.858 | 0.670 | 0.920 | 0.956 | 2 |
| SIP-jmed-13B | medical | 4096 | 0.894 | [0.878, 0.909] | 0.906 | 0.882 | 0.856 | [0.837, 0.874] | 0.889 | 0.873 | 0.842 | 0.657 | 0.907 | 0.953 | 3 |
| LLM-jp-3.1-13B (base of SIP-jmed) | general | 2600 | 0.878 | [0.855, 0.899] | 0.899 | 0.859 | 0.834 | [0.807, 0.858] | 0.873 | 0.854 | 0.820 | 0.601 | 0.890 | 0.945 | 6 |
| Qwen3-30B-A3B (MoE) | general | 4096 | 0.884 | [0.866, 0.899] | 0.893 | 0.875 | 0.841 | [0.820, 0.859] | 0.877 | 0.861 | 0.822 | 0.618 | 0.898 | 0.946 | 3 |
| LLM-jp-4-32B (MoE, reasoning) | general | 4096 | 0.881 | [0.865, 0.895] | 0.888 | 0.873 | 0.835 | [0.816, 0.853] | 0.874 | 0.856 | 0.819 | 0.618 | 0.893 | 0.942 | 2 |
| LLM-jp-1.8B | general | 2600 | 0.830 | [0.805, 0.854] | 0.863 | 0.800 | 0.776 | [0.749, 0.801] | 0.822 | 0.802 | 0.763 | 0.549 | 0.844 | 0.928 | 10 |
| UTH-BERT v2 (retrained, seed 42) | medical | – | 0.832 | [0.794, 0.863] | 0.843 | 0.822 | 0.759 | [0.722, 0.789] | 0.816 | 0.782 | 0.808 | 0.545 | 0.847 | 0.902 | – |
| NICT-BERT v2 (retrained, seed 42) | general | – | 0.841 | [0.806, 0.870] | 0.841 | 0.841 | 0.805 | [0.767, 0.837] | 0.836 | 0.820 | 0.803 | 0.598 | 0.856 | 0.952 | – |
| UTH-BERT (original training, coverage completed) | medical | – | 0.834 | [0.796, 0.864] | 0.843 | 0.825 | 0.761 | [0.725, 0.792] | 0.819 | 0.785 | 0.818 | 0.589 | 0.846 | 0.903 | – |
| NICT-BERT (original training, coverage completed) | general | – | 0.828 | [0.791, 0.858] | 0.826 | 0.830 | 0.787 | [0.748, 0.821] | 0.823 | 0.805 | 0.788 | 0.564 | 0.840 | 0.943 | – |

#### Table A2. Fine-tuned models, LAU setting (generation budget 900 tokens), seed 42

95% CI: document-level bootstrap (36 documents, 2,000 resamples). min2/min3: relaxed F1 with a minimum substring length of 2/3 characters. Macro: mean F1 over the 48 labels with ≥ 20 test entities. Merged: relaxed F1 after merging the `-patient`/`-others` suffixes. Exact/TP: share of relaxed true positives whose surface is an exact match. Trunc.: sentences cut off at the generation budget.

| Model | Type | Budget | Relaxed F1 | 95% CI | P | R | Strict F1 | 95% CI | min2 | min3 | Macro48 | Macro98 | Merged | Exact/TP | Trunc./433 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | medical | 900 | 0.825 | [0.787, 0.858] | 0.924 | 0.745 | 0.793 | [0.756, 0.826] | 0.820 | 0.807 | 0.770 | 0.551 | 0.837 | 0.958 | 87 |
| Llama-3.3-70B | general | 900 | 0.823 | [0.786, 0.854] | 0.921 | 0.744 | 0.791 | [0.754, 0.823] | 0.818 | 0.806 | 0.777 | 0.562 | 0.834 | 0.958 | 86 |
| MedGemma-27B | medical | 900 | 0.824 | [0.786, 0.856] | 0.918 | 0.747 | 0.790 | [0.752, 0.823] | 0.818 | 0.804 | 0.775 | 0.557 | 0.836 | 0.956 | 81 |
| Gemma-3-27B (base of MedGemma) | general | 900 | 0.828 | [0.792, 0.860] | 0.923 | 0.751 | 0.796 | [0.759, 0.829] | 0.823 | 0.809 | 0.779 | 0.558 | 0.838 | 0.958 | 84 |
| SIP-jmed-13B | medical | 900 | 0.820 | [0.782, 0.853] | 0.919 | 0.741 | 0.787 | [0.749, 0.820] | 0.816 | 0.802 | 0.772 | 0.570 | 0.830 | 0.956 | 84 |
| LLM-jp-3.1-13B (base of SIP-jmed) | general | 900 | 0.816 | [0.779, 0.848] | 0.913 | 0.737 | 0.777 | [0.740, 0.810] | 0.811 | 0.795 | 0.763 | 0.550 | 0.826 | 0.948 | 82 |
| Qwen3-30B-A3B (MoE) | general | 900 | 0.812 | [0.774, 0.845] | 0.912 | 0.732 | 0.774 | [0.736, 0.807] | 0.805 | 0.790 | 0.760 | 0.536 | 0.822 | 0.951 | 85 |
| LLM-jp-4-32B (MoE, reasoning) | general | 900 | 0.813 | [0.776, 0.843] | 0.904 | 0.738 | 0.772 | [0.735, 0.803] | 0.807 | 0.790 | 0.758 | 0.531 | 0.823 | 0.946 | 79 |
| LLM-jp-1.8B | general | 900 | 0.777 | [0.739, 0.810] | 0.880 | 0.695 | 0.730 | [0.691, 0.765] | 0.770 | 0.752 | 0.716 | 0.496 | 0.789 | 0.935 | 81 |
| UTH-BERT v2 (retrained, seed 42) | medical | – | 0.832 | [0.794, 0.863] | 0.843 | 0.822 | 0.759 | [0.722, 0.789] | 0.816 | 0.782 | 0.808 | 0.545 | 0.847 | 0.902 | – |
| NICT-BERT v2 (retrained, seed 42) | general | – | 0.841 | [0.806, 0.870] | 0.841 | 0.841 | 0.805 | [0.767, 0.837] | 0.836 | 0.820 | 0.803 | 0.598 | 0.856 | 0.952 | – |
| UTH-BERT (original training, coverage completed) | medical | – | 0.834 | [0.796, 0.864] | 0.843 | 0.825 | 0.761 | [0.725, 0.792] | 0.819 | 0.785 | 0.818 | 0.589 | 0.846 | 0.903 | – |
| NICT-BERT (original training, coverage completed) | general | – | 0.828 | [0.791, 0.858] | 0.826 | 0.830 | 0.787 | [0.748, 0.821] | 0.823 | 0.805 | 0.788 | 0.564 | 0.840 | 0.943 | – |

#### Table B. Three QLoRA seeds (42 / 1 / 2), main condition (4,096-token budget)

| Model | Type | Relaxed 42 / 1 / 2 | Mean ± SD | Strict 42 / 1 / 2 | Mean ± SD | Macro (mean) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | medical | 0.905 / 0.908 / 0.912 | 0.908 ± 0.003 | 0.869 / 0.874 / 0.879 | 0.874 ± 0.005 | 0.852 |
| Llama-3.3-70B | general | 0.909 / 0.912 / 0.916 | 0.912 ± 0.004 | 0.871 / 0.879 / 0.883 | 0.878 ± 0.006 | 0.863 |
| MedGemma-27B | medical | 0.899 / 0.909 / 0.909 | 0.906 ± 0.006 | 0.859 / 0.873 / 0.873 | 0.868 ± 0.008 | 0.855 |
| Gemma-3-27B (base of MedGemma) | general | 0.909 / 0.903 / 0.903 | 0.905 ± 0.004 | 0.873 / 0.867 / 0.867 | 0.869 ± 0.004 | 0.853 |
| SIP-jmed-13B | medical | 0.894 / 0.899 / 0.899 | 0.897 ± 0.003 | 0.856 / 0.857 / 0.857 | 0.857 ± 0.001 | 0.851 |
| LLM-jp-3.1-13B (base of SIP-jmed) | general | 0.878 / 0.882 / 0.880 | 0.880 ± 0.002 | 0.834 / 0.841 / 0.838 | 0.838 ± 0.003 | 0.824 |
| Qwen3-30B-A3B (MoE) | general | 0.884 / 0.894 / 0.895 | 0.891 ± 0.006 | 0.841 / 0.854 / 0.853 | 0.850 ± 0.007 | 0.833 |
| LLM-jp-4-32B (MoE, reasoning) | general | 0.881 / 0.892 / 0.891 | 0.888 ± 0.006 | 0.835 / 0.849 / 0.849 | 0.844 ± 0.008 | 0.832 |
| LLM-jp-1.8B | general | 0.830 / 0.828 / 0.834 | 0.831 ± 0.003 | 0.776 / 0.776 / 0.780 | 0.777 ± 0.002 | 0.769 |
| UTH-BERT v2 (retrained) | medical | 0.832 / 0.836 / 0.840 | 0.836 ± 0.004 | 0.759 / 0.765 / 0.768 | 0.764 ± 0.005 | 0.815 |
| NICT-BERT v2 (retrained) | general | 0.841 / 0.840 / 0.841 | 0.841 ± 0.001 | 0.805 / 0.802 / 0.802 | 0.803 ± 0.002 | 0.798 |

#### Table B2. Three QLoRA seeds (42 / 1 / 2), LAU setting (900-token budget)

| Model | Type | Relaxed 42 / 1 / 2 | Mean ± SD | Strict 42 / 1 / 2 | Mean ± SD | Macro (mean) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | medical | 0.825 / 0.826 / 0.829 | 0.827 ± 0.002 | 0.793 / 0.798 / 0.802 | 0.798 ± 0.004 | 0.773 |
| Llama-3.3-70B | general | 0.823 / 0.828 / 0.830 | 0.827 ± 0.003 | 0.791 / 0.801 / 0.801 | 0.798 ± 0.006 | 0.778 |
| MedGemma-27B | medical | 0.824 / 0.829 / 0.830 | 0.827 ± 0.003 | 0.790 / 0.797 / 0.798 | 0.795 ± 0.004 | 0.780 |
| Gemma-3-27B (base of MedGemma) | general | 0.828 / 0.825 / 0.828 | 0.827 ± 0.002 | 0.796 / 0.793 / 0.798 | 0.796 ± 0.002 | 0.777 |
| SIP-jmed-13B | medical | 0.820 / 0.819 / 0.817 | 0.819 ± 0.002 | 0.787 / 0.785 / 0.783 | 0.785 ± 0.002 | 0.771 |
| LLM-jp-3.1-13B (base of SIP-jmed) | general | 0.816 / 0.817 / 0.817 | 0.817 ± 0.001 | 0.777 / 0.781 / 0.781 | 0.780 ± 0.002 | 0.765 |
| Qwen3-30B-A3B (MoE) | general | 0.812 / 0.815 / 0.817 | 0.815 ± 0.002 | 0.774 / 0.781 / 0.782 | 0.779 ± 0.004 | 0.763 |
| LLM-jp-4-32B (MoE, reasoning) | general | 0.813 / 0.820 / 0.819 | 0.817 ± 0.004 | 0.772 / 0.783 / 0.783 | 0.779 ± 0.006 | 0.765 |
| LLM-jp-1.8B | general | 0.777 / 0.778 / 0.780 | 0.778 ± 0.002 | 0.730 / 0.730 / 0.733 | 0.731 ± 0.002 | 0.724 |

#### Table C. Generation-budget ablation: 900 → 4,096 tokens (seed 42; last column: three-seed means)

| Model | Budget | Relaxed F1 900 → large | Δ | Strict F1 900 → large | R 900 → large | P 900 → large | Trunc. 900 → large | 3-seed relaxed 900 → large |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | 4096 | 0.825 → 0.905 | +0.080 | 0.793 → 0.869 | 0.745 → 0.899 | 0.924 → 0.912 | 87 → 2 | 0.827 → 0.908 |
| Llama-3.3-70B | 4096 | 0.823 → 0.909 | +0.086 | 0.791 → 0.871 | 0.744 → 0.902 | 0.921 → 0.915 | 86 → 2 | 0.827 → 0.912 |
| MedGemma-27B | 4096 | 0.824 → 0.899 | +0.076 | 0.790 → 0.859 | 0.747 → 0.893 | 0.918 → 0.906 | 81 → 2 | 0.827 → 0.906 |
| Gemma-3-27B (base of MedGemma) | 4096 | 0.828 → 0.909 | +0.081 | 0.796 → 0.873 | 0.751 → 0.905 | 0.923 → 0.914 | 84 → 2 | 0.827 → 0.905 |
| SIP-jmed-13B | 4096 | 0.820 → 0.894 | +0.074 | 0.787 → 0.856 | 0.741 → 0.882 | 0.919 → 0.906 | 84 → 3 | 0.819 → 0.897 |
| LLM-jp-3.1-13B (base of SIP-jmed) | 2600 | 0.816 → 0.878 | +0.062 | 0.777 → 0.834 | 0.737 → 0.859 | 0.913 → 0.899 | 82 → 6 | 0.817 → 0.880 |
| Qwen3-30B-A3B (MoE) | 4096 | 0.812 → 0.884 | +0.072 | 0.774 → 0.841 | 0.732 → 0.875 | 0.912 → 0.893 | 85 → 3 | 0.815 → 0.891 |
| LLM-jp-4-32B (MoE, reasoning) | 4096 | 0.813 → 0.881 | +0.068 | 0.772 → 0.835 | 0.738 → 0.873 | 0.904 → 0.888 | 79 → 2 | 0.817 → 0.888 |
| LLM-jp-1.8B | 2600 | 0.777 → 0.830 | +0.053 | 0.730 → 0.776 | 0.695 → 0.800 | 0.880 → 0.863 | 81 → 10 | 0.778 → 0.831 |

#### Table D. Zero-shot: prompt only vs. JSON-schema constrained decoding (relaxed / strict F1; 900-token budget, 3,072 for reasoning models)

Parser v1 = the LAU-version parser (first `[` … last `]`); parser v2 = last complete JSON array (reasoning channel discarded). Fail: sentences without a parseable array. Unk.: predictions with a label outside the 109-label list. Trunc.: sentences cut off by the generation budget.

| Model | ZS F1 (v1) | Fail (v1) | Unk. (v1) | ZS F1 (v2) rel. / str. | Fail (v2) | Trunc. | ZS+constrained rel. / str. | P / R | Trunc. | FT F1 (main, s42) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | 0.136 | 10 | 581 | 0.136 / 0.044 | 6 | 1 | 0.146 / 0.044 | 0.345 / 0.093 | 0 | 0.905 |
| Llama-3.3-70B | 0.276 | 0 | 188 | 0.276 / 0.160 | 0 | 2 | 0.265 / 0.157 | 0.508 / 0.179 | 3 | 0.909 |
| MedGemma-27B | 0.320 | 0 | 197 | 0.320 / 0.209 | 0 | 40 | 0.316 / 0.205 | 0.452 / 0.243 | 28 | 0.899 |
| SIP-jmed-13B | 0.010 | 322 | 47 | 0.025 / 0.014 | 92 | 147 | 0.015 / 0.003 | 0.043 / 0.009 | 108 | 0.894 |
| Qwen3-30B-A3B (MoE) | 0.235 | 3 | 236 | 0.235 / 0.139 | 3 | 24 | 0.224 / 0.133 | 0.332 / 0.169 | 46 | 0.884 |
| LLM-jp-4-32B (MoE, reasoning) | 0.092 | 261 | 95 | 0.150 / 0.068 | 17 | 35 | 0.000 / 0.000 | 0.000 / 0.000 | 1 | 0.881 |
| LLM-jp-1.8B | 0.027 | 3 | 195 | 0.027 / 0.001 | 3 | 427 | 0.019 / 0.001 | 0.044 / 0.013 | 343 | 0.830 |

#### Table E. Reasoning models: four zero-shot conditions (relaxed F1)

| Model | Prompt only, parser v1 | Prompt only, parser v2 | Whole-output constraint | Think, then constrain (rel. / str.) | Fine-tuned |
|---|:-:|:-:|:-:|:-:|:-:|
| LLM-jp-4-32B | 0.092 (fail 261) | 0.150 (fail 17) | 0.000 | 0.172 / 0.081 | 0.881 |
| SIP-jmed-13B | 0.010 (fail 322) | 0.025 (fail 92) | 0.015 | 0.048 / 0.023 | 0.894 |

#### Table F. Reference models (ollama; cannot be fine-tuned locally)

| Model | Condition | Relaxed F1 | Strict F1 | P / R | Fail | Trunc. |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Qwen3.6-35B | prompt only, reasoning off | 0.300 | 0.153 | 0.439 / 0.228 | 0 | 3 |
| Qwen3.6-35B | reasoning on + schema on answer (3,072-token budget) | 0.022 | 0.017 | 0.561 / 0.011 | 310 | 310 |
| GPT-OSS-120B | prompt only (reasoning cannot be disabled) | 0.265 | 0.131 | 0.571 / 0.173 | 8 | 16 |
| GPT-OSS-120B | reasoning + schema on answer | 0.263 | 0.135 | 0.562 / 0.172 | 11 | 11 |

#### Table G. Medical model minus its general-purpose base, same seed and same generation budget (paired bootstrap, relaxed F1), main condition (4,096 tokens; the 13B pair at 2,600 tokens, the context limit of the base model)

| Pair | Seed 42 | Seed 1 | Seed 2 |
|---|:-:|:-:|:-:|
| SIP-jmed-13B − LLM-jp-3.1-13B | +0.007 [+0.000, +0.015] | +0.007 [+0.002, +0.013] | +0.008 [+0.001, +0.015] |
| MedGemma-27B − Gemma-3-27B | -0.010 [-0.016, -0.004] | +0.006 [+0.001, +0.011] | +0.007 [-0.002, +0.020] |
| Med42-70B − Llama-3.3-70B | -0.003 [-0.010, +0.002] | -0.004 [-0.010, +0.003] | -0.004 [-0.009, +0.000] |

#### Table G2. Same, 900-token budget

| Pair | Seed 42 | Seed 1 | Seed 2 |
|---|:-:|:-:|:-:|
| SIP-jmed-13B − LLM-jp-3.1-13B | +0.004 [+0.000, +0.009] | +0.002 [-0.001, +0.006] | +0.000 [-0.007, +0.006] |
| MedGemma-27B − Gemma-3-27B | -0.005 [-0.010, +0.000] | +0.004 [-0.002, +0.010] | +0.001 [-0.002, +0.005] |
| Med42-70B − Llama-3.3-70B | +0.002 [-0.002, +0.006] | -0.001 [-0.005, +0.003] | -0.001 [-0.005, +0.003] |

#### Table H. Fine-tuned LLM minus BERT, seed 42 (paired bootstrap, relaxed F1): main condition / LAU setting

| LLM | vs. UTH-BERT (4,096) | vs. NICT-BERT (4,096) | vs. UTH-BERT (900) | vs. NICT-BERT (900) |
|---|:-:|:-:|:-:|:-:|
| Med42-70B | +0.073 [+0.052, +0.103] | +0.065 [+0.044, +0.094] | -0.007 [-0.038, +0.023] | -0.016 [-0.046, +0.013] |
| Llama-3.3-70B | +0.076 [+0.052, +0.111] | +0.068 [+0.045, +0.100] | -0.009 [-0.042, +0.024] | -0.018 [-0.048, +0.014] |
| MedGemma-27B | +0.067 [+0.046, +0.097] | +0.059 [+0.038, +0.087] | -0.009 [-0.041, +0.025] | -0.017 [-0.049, +0.013] |
| Gemma-3-27B (base of MedGemma) | +0.077 [+0.053, +0.112] | +0.068 [+0.045, +0.100] | -0.004 [-0.037, +0.030] | -0.013 [-0.044, +0.019] |
| SIP-jmed-13B | +0.062 [+0.039, +0.093] | +0.053 [+0.032, +0.083] | -0.012 [-0.045, +0.021] | -0.021 [-0.052, +0.010] |
| LLM-jp-3.1-13B (base of SIP-jmed) | +0.046 [+0.023, +0.076] | +0.037 [+0.015, +0.065] | -0.016 [-0.049, +0.017] | -0.025 [-0.056, +0.006] |
| Qwen3-30B-A3B (MoE) | +0.052 [+0.028, +0.086] | +0.043 [+0.021, +0.075] | -0.020 [-0.054, +0.014] | -0.029 [-0.060, +0.002] |
| LLM-jp-4-32B (MoE, reasoning) | +0.048 [+0.026, +0.081] | +0.040 [+0.018, +0.070] | -0.020 [-0.050, +0.012] | -0.028 [-0.058, +0.001] |
| LLM-jp-1.8B | -0.002 [-0.026, +0.027] | -0.011 [-0.034, +0.017] | -0.055 [-0.088, -0.023] | -0.064 [-0.095, -0.033] |

#### Table I. Effect of the 900-token generation budget (fine-tuned, seed 42, relaxed)

Trunc.: sentences whose output was cut off; Gold: share of all gold entities in those sentences; R: recall on truncated / other sentences; FN: share of the model's false negatives in truncated sentences. UTH-BERT is scored on the sentences truncated by Med42-70B.

| Model | Trunc. | Gold share | R (trunc.) | R (rest) | P (trunc.) | P (rest) | FN in trunc. |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | 87 | 0.53 | 0.583 | 0.927 | 0.908 | 0.937 | 0.87 |
| Llama-3.3-70B | 86 | 0.53 | 0.580 | 0.925 | 0.909 | 0.930 | 0.86 |
| MedGemma-27B | 81 | 0.51 | 0.577 | 0.920 | 0.901 | 0.930 | 0.84 |
| Gemma-3-27B | 84 | 0.52 | 0.587 | 0.927 | 0.905 | 0.936 | 0.86 |
| SIP-jmed-13B | 84 | 0.52 | 0.578 | 0.916 | 0.904 | 0.929 | 0.84 |
| LLM-jp-3.1-13B | 82 | 0.51 | 0.572 | 0.909 | 0.896 | 0.925 | 0.83 |
| Qwen3-30B-A3B | 85 | 0.52 | 0.571 | 0.907 | 0.892 | 0.926 | 0.83 |
| LLM-jp-4-32B | 79 | 0.50 | 0.571 | 0.903 | 0.886 | 0.916 | 0.81 |
| LLM-jp-1.8B | 81 | 0.50 | 0.525 | 0.867 | 0.852 | 0.898 | 0.78 |
| UTH-BERT (same split as Med42) | (87) | 0.53 | 0.798 | 0.848 | 0.812 | 0.878 | – |

#### Table J. Per-label scores of fine-tuned Med42-70B (seed 42, 4,096-token budget) for nine labels; UTH = UTH-BERT relaxed / strict F1

| Label | n | P | R | F1 (relaxed) | F1 (strict) | UTH (rel. / str.) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| state-patient | 2318 | 0.91 | 0.92 | 0.91 | 0.85 | 0.79 / 0.71 |
| clinical_test | 587 | 0.94 | 0.93 | 0.93 | 0.88 | 0.81 / 0.74 |
| treatment | 614 | 0.91 | 0.91 | 0.91 | 0.84 | 0.85 / 0.79 |
| drug | 283 | 0.95 | 0.95 | 0.95 | 0.94 | 0.81 / 0.77 |
| PN-Negative-patient | 276 | 0.94 | 0.81 | 0.87 | 0.84 | 0.94 / 0.90 |
| state-others | 155 | 0.89 | 0.41 | 0.56 | 0.53 | 0.45 / 0.40 |
| device | 75 | 0.76 | 0.85 | 0.81 | 0.65 | 0.58 / 0.52 |
| time_span | 59 | 0.73 | 0.63 | 0.67 | 0.65 | 0.64 / 0.58 |
| value-others | 63 | 0.38 | 0.22 | 0.28 | 0.28 | 0.29 / 0.26 |

#### Table K. Output completeness of fine-tuned models (seed 42): 900-token budget / main condition

Stop: sentences where generation ended with the end-of-sequence token; Clean: raw output is itself a valid JSON array (Harmony channel markers stripped for LLM-jp-4); Repaired: array cut off by the budget and closed at the last complete element; Continued: output continued after a complete array (parsed as the first array); Empty: parsed answer has no entities; Fail: no array could be parsed.

| Model | Stop (900 / main) | Clean (900 / main) | Repaired (900 / main) | Continued (900 / main) | Empty (900 / main) | Fail (900 / main) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | 346 / 431 | 346 / 431 | 87 / 2 | 0 / 0 | 0 / 0 | 0 / 0 |
| Llama-3.3-70B | 347 / 431 | 347 / 431 | 86 / 2 | 0 / 0 | 0 / 0 | 0 / 0 |
| MedGemma-27B | 352 / 431 | 352 / 427 | 81 / 2 | 0 / 4 | 0 / 0 | 0 / 0 |
| Gemma-3-27B (base of MedGemma) | 349 / 431 | 349 / 431 | 84 / 2 | 0 / 0 | 0 / 0 | 0 / 0 |
| SIP-jmed-13B | 349 / 430 | 351 / 428 | 81 / 3 | 0 / 0 | 0 / 1 | 0 / 0 |
| LLM-jp-3.1-13B (base of SIP-jmed) | 351 / 427 | 351 / 427 | 82 / 6 | 0 / 0 | 0 / 0 | 0 / 0 |
| Qwen3-30B-A3B (MoE) | 348 / 430 | 347 / 429 | 85 / 3 | 0 / 0 | 1 / 1 | 0 / 0 |
| LLM-jp-4-32B (MoE, reasoning) | 354 / 431 | 355 / 431 | 78 / 2 | 0 / 0 | 0 / 0 | 0 / 0 |
| LLM-jp-1.8B | 352 / 423 | 352 / 420 | 80 / 10 | 0 / 0 | 2 / 2 | 0 / 0 |

Spearman correlation between label frequency and relaxed F1 (Med42-70B, 98 labels): ρ = 0.43 (p = 1e-05). The 27 `-others` labels average F1 0.23.
