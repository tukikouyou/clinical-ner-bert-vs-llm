import os
import pandas as pd
import numpy as np


# 固有表現とその開始位置と終了位置を得る
def list2taple(labels):
    tmp = []
    seq_length = len(labels)
    for i in range(0, seq_length, 1):
        # 固有表現の場合
        if labels[i].startswith("B-"):
            start_tag = labels[i][2:]
            # iが最後のindexの場合 (= len(seq_length)-1)
            if i == seq_length-1:
                tmp.append((i, i, start_tag))
                continue
            for j in range(i+1, seq_length, 1):
                inside_tag = labels[j]
                # jが最後の単語indexの場合
                if j == seq_length-1 and inside_tag == "I-{0}".format(start_tag):
                    tmp.append((i, j, start_tag))
                    break
                # jが最後以外のindexでかつI-{tag}の場合
                elif inside_tag == "I-{0}".format(start_tag):
                    pass
                #  I-{tag}ではない場合
                else:
                    tmp.append((i, j-1, start_tag))
                    break
    return tmp



def calculate_precision_recall_f1_from_triplet(golds_set, preds_set):
    # 集計
    tp = len(preds_set & golds_set)
    fp = len(preds_set - golds_set)
    fn = len(golds_set - preds_set)
    # micro avgに相当
    if (fp + tp) == 0:
        precision = 0
    else:
        precision = tp / (fp + tp)
    if (fn + tp) == 0:
        recall = 0
    else:
        recall = tp / (fn + tp)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, tp, fp, fn


def calculate_precision_recall_f1_from_count(tp, fp, fn):
    # micro avgに相当
    if (fp + tp) == 0:
        precision = 0
    else:
        precision = tp / (fp + tp)
    if (fn + tp) == 0:
        recall = 0
    else:
        recall = tp / (fn + tp)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def output_dict(res_dct):
    score_dct = {}
    sum_tp, sum_fp, sum_fn = 0, 0, 0
    for key in res_dct.keys():
        socre = res_dct[key]
        tp, fp, fn = socre["tp"], socre["fp"], socre["fn"]
        # micro avgに相当
        if (fp + tp) == 0:
            precision = 0
        else:
            precision = tp / (fp + tp)
        if (fn + tp) == 0:
            recall = 0
        else:
            recall = tp / (fn + tp)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        score_dct[key] = {"precision": precision, "recall": recall, "f1": f1, "support":tp + fn}
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
    precision, recall, f1 = calculate_precision_recall_f1_from_count(sum_tp, sum_fp, sum_fn)
    # Macro
    #p = sum([score_dct[key]["precision"] for key in score_dct.keys()])/len([score_dct[key]["precision"] for key in score_dct.keys()])
    #r = sum([score_dct[key]["recall"] for key in score_dct.keys()])/len([score_dct[key]["recall"] for key in score_dct.keys()])
    #f = sum([score_dct[key]["f1"] for key in score_dct.keys()])/len([score_dct[key]["f1"] for key in score_dct.keys()])

    p = np.array([score_dct[key]["precision"] for key in score_dct.keys()]).mean()
    r = np.array([score_dct[key]["recall"] for key in score_dct.keys()]).mean()
    f = np.array([score_dct[key]["f1"] for key in score_dct.keys()]).mean()

    score_dct["macro avg"] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}
    score_dct["macro avg"]["precision"] = p
    score_dct["macro avg"]["recall"] = r
    score_dct["macro avg"]["f1"] = f
    score_dct["macro avg"]["support"] = sum_tp + sum_fn
    # Micro
    score_dct["micro avg"] = {"precision": precision, "recall": recall, "f1": f1, "support":sum_tp + sum_fn}
    return score_dct


# StrictなNERの評価
def eval_ner_strict(df):
    # 全てのタグ
    all_tags = list(set([x[2:] for x in df["IOB"] if x != "O"] + [x[2:] for x in df["pred_IOB"] if x != "O"]))
    # それぞれのタグのTP, FP, FNを代入する辞書
    res_dct = {x: {"tp": 0, "fp": 0, "fn": 0} for x in all_tags}
    # 症例ごとに処理（indexとserialを修正すればまとめてできる？）
    for ids in df["unique_no"].unique():
        tmp_df = df[df["unique_no"]==ids]
        #serial2index = {s: i for s, i in zip(tmp_df["serial"], tmp_df["index"]) if i != -999}
        #index2serial = {i: s for s, i in zip(tmp_df["serial"], tmp_df["index"]) if i != -999}
        # タグのリストを得る
        gold_tags = tmp_df["IOB"].to_list()
        pred_tags = tmp_df["pred_IOB"].to_list()
        # (開始位置、終了位置、タグ)のトリプレットを作成
        # 位置は形態素数でカウント e.g. 私 は 田中で田中の開始位置は2、終了位置も2
        golds_taple = list2taple(gold_tags)
        preds_taple = list2taple(pred_tags)
        # 集合に変換
        golds_set = {x for x in golds_taple}
        preds_set = {x for x in preds_taple}
        # タグごとの計算
        for tag in all_tags:
            # 正解
            golds = set([])
            for g in golds_set:
                if g[2] == tag:
                    golds.add(g)
            # 予測
            preds = set([])
            for p in preds_set:
                if p[2] == tag:
                    preds.add(p)
            # 評価
            _, _, _, tp, fp, fn = calculate_precision_recall_f1_from_triplet(golds, preds)
            res_dct[tag]["tp"] += tp
            res_dct[tag]["fp"] += fp
            res_dct[tag]["fn"] += fn
    # スコアを計算
    score_dct = output_dict(res_dct)
    return score_dct


# SoftなNERの評価
def eval_ner_soft(df):
    #list_df = pd.concat([df[df["unique_no"]==ids] for ids in df["unique_no"].unique()])
    all_tags = list(set([x[2:] for x in df["IOB"] if x != "O"] + [x[2:] for x in df["pred_IOB"] if x != "O"]))
    res_dct = {x: {"tp": 0, "fp": 0, "fn": 0} for x in all_tags}
    for ids in df["unique_no"].unique():
        tmp_df = df[df["unique_no"]==ids]
        #serial2index = {s: i for s, i in zip(tmp_df["serial"], tmp_df["index"]) if i != -999}
        #index2serial = {i: s for s, i in zip(tmp_df["serial"], tmp_df["index"]) if i != -999}
        gold_tags = tmp_df["IOB"].to_list()
        pred_tags = tmp_df["pred_IOB"].to_list()
        golds_taple = list2taple(gold_tags)
        preds_taple = list2taple(pred_tags)
        # 固有表現抽出の評価（全部）
        golds_set = {x for x in golds_taple}
        preds_set = {x for x in preds_taple}
        # Softに変更
        mod_preds_set = set([])
        for pred in preds_set:
            tmp = 0
            # 区間を求める
            for gold in golds_set:
                # equal
                if gold == pred:
                    #print((gold, pred, "equal"))
                    mod_preds_set.add(gold)
                    tmp = 1
                # start
                elif gold[0] == pred[0] and gold[2] == pred[2] and gold[1] > pred[1]:
                    #print((gold, pred, "start"))
                    mod_preds_set.add(gold)
                    tmp = 1
                # finish
                elif gold[0] < pred[0] and gold[1] == pred[1] and gold[2] == pred[2]:
                    #print((gold, pred, "finish"))
                    mod_preds_set.add(gold)
                    tmp = 1
                # contain
                elif gold[0] < pred[0] and gold[1] > pred[1] and gold[2] == pred[2]:
                    #print((gold, pred, "contain"))
                    mod_preds_set.add(gold)
                    tmp = 1
                # during
                elif gold[0] > pred[0] and gold[1] < pred[1] and gold[2] == pred[2]:
                    #print((gold, pred, "during"))
                    mod_preds_set.add(gold)
                    tmp = 1
                # overlap
                #始点2 <= 終点1 && 始点1 <= 終点2
                elif gold[0] <= pred[1] and pred[0] <= gold[1] and gold[2] == pred[2]:
                    #print((gold, pred, "overlap"))
                    mod_preds_set.add(gold)
                    tmp = 1
                else:
                    pass
            if tmp == 0:
                #print(((-999, -999, "None"), pred, "None"))
                mod_preds_set.add(pred)
        # タグごとの計算
        for tag in all_tags:
            # 正解
            golds = set([])
            for g in golds_set:
                if g[2] == tag:
                    golds.add(g)
            # 予測
            preds = set([])
            for p in mod_preds_set:
                if p[2] == tag:
                    preds.add(p)
            # 評価
            precision, recall, f1, tp, fp, fn = calculate_precision_recall_f1_from_triplet(golds, preds)
            res_dct[tag]["tp"] += tp
            res_dct[tag]["fp"] += fp
            res_dct[tag]["fn"] += fn
    score_dct_soft = output_dict(res_dct)
    return score_dct_soft


# Strictな関係の評価
