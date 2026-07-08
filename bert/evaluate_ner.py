# -*- coding: utf-8 -*-
"""
NER予測結果(gold IOB + pred_IOB 付きCSV)の評価スクリプト。

NER_training.py が保存する ./results/NER/<bert_type>/NER/<fold>.csv のように、
IOB(正解)とpred_IOB(予測)の両列を持つCSVから strict / soft のspan-level
P/R/F1 をタグ別に計算してJSON保存する。

使い方:
    python3 evaluate_ner.py --pred_csv ./results/NER/UTH/NER/0.csv \
        --output ./results/NER/UTH/NER/eval_0.json
"""
import json
import argparse
import pandas as pd
from lib.util import eval_ner_strict, eval_ner_soft
from lib.util.utils import NpEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_csv', type=str, required=True,
                        help="IOB(正解)とpred_IOB(予測)を含むCSV")
    parser.add_argument('--output', type=str, default=None,
                        help="評価結果JSONの保存先(省略時は<pred_csv>_eval.json)")
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    for col in ("IOB", "pred_IOB"):
        if col not in df.columns:
            raise SystemExit(f"エラー: 列 {col} がありません({args.pred_csv})")

    res = {"strict": eval_ner_strict(df), "soft": eval_ner_soft(df)}

    # マイクロ/マクロ平均を表示
    for mode in ("strict", "soft"):
        for avg in ("micro avg", "macro avg"):
            s = res[mode][avg]
            print(f"[{mode}] {avg}: P={s['precision']:.4f} R={s['recall']:.4f} "
                  f"F1={s['f1']:.4f} (support={s['support']})")

    out_path = args.output or args.pred_csv.rsplit(".", 1)[0] + "_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4, ensure_ascii=False, cls=NpEncoder)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
