# -*- coding: utf-8 -*-
"""
4条件(UTH/NICT × finetune/no-finetune)の予測結果と、
ファインチューニング済みモデルのテスト評価をまとめて比較表示。

使い方:
    python3 summarize_results.py
"""
import os
import json


def load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


CONFIGS = [
    ("UTH  (finetuned)", "results/predict/UTH", "results/NER/UTH/NER/metrics_0.json"),
    ("NICT (finetuned)", "results/predict/NICT", "results/NER/NICT/NER/metrics_0.json"),
    ("UTH  (no-finetune)", "results/predict/UTH_nofinetune", None),
    ("NICT (no-finetune)", "results/predict/NICT_nofinetune", None),
]


def main():
    print("=" * 78)
    print("ファインチューニング済みモデルのテスト精度 (iCorpus テスト分割, gold有)")
    print("=" * 78)
    hdr = f"{'model':<20} {'strict-F1':>10} {'soft-F1':>9} {'best_ep':>8} {'train_s':>8}"
    print(hdr); print("-" * len(hdr))
    for name, _, mpath in CONFIGS:
        m = load(mpath) if mpath else None
        if m:
            print(f"{name:<20} {m['test_strict_micro']['f1']:>10.4f} "
                  f"{m['test_soft_micro']['f1']:>9.4f} {str(m['best_epoch']):>8} "
                  f"{str(m['train_total_sec']):>8}")
        else:
            print(f"{name:<20} {'—':>10} {'—':>9} {'—':>8} {'—':>8}  (NERヘッド未学習)")

    print("\n" + "=" * 78)
    print("実データ(電子カルテ 46,668報告 / 93,110テキスト)からの抽出統計")
    print("=" * 78)
    hdr2 = f"{'model':<20} {'n_entities':>12} {'/text':>7} {'空予測chunk':>10} {'秒':>7}"
    print(hdr2); print("-" * len(hdr2))
    details = []
    for name, ppath, _ in CONFIGS:
        s = load(os.path.join(ppath, "stats.json"))
        if not s:
            print(f"{name:<20} {'(未実行)':>12}")
            continue
        print(f"{name:<20} {s['n_entities']:>12,} {s['entities_per_text']:>7.1f} "
              f"{s['n_empty_chunks']:>10,} {s['elapsed_sec']:>7.0f}")
        details.append((name, s))

    print("\n--- 上位ラベル(件数) ---")
    for name, s in details:
        top = list(s["per_label"].items())[:8]
        print(f"[{name}]")
        print("   " + ", ".join(f"{k}={v:,}" for k, v in top))

    print("\n注: no-finetune は事前学習BERT本体+未学習(ランダム)NERヘッドの下限ベースライン。")
    print("    テスト分割のgold評価では strict-F1 ≒ 0 (実質ランダム)。")


if __name__ == "__main__":
    main()
