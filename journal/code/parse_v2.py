# -*- coding: utf-8 -*-
"""改良パーサ v2: 出力中の「最後の整形 JSON 配列」を取り出す。
元の parse_entities (LAU 論文) は最初の '[' から最後の ']' までを貪欲に取るため、
思考文中に '[' があると最終回答の JSON があっても失敗する(LLM-jp-4 で 242/433 句)。
v2: (1) Harmony の final チャネル以降を優先、(2) 右端から '[' を走査し raw_decode で
    list[dict(label,text)] になる最後の配列を採用、(3) 無ければ元パーサ(截断修復込み)にフォールバック。"""
import re, json
_dec = json.JSONDecoder()


def _valid(o):
    return isinstance(o, list) and all(isinstance(x, dict) and "label" in x and "text" in x for x in o)


def parse_entities_v2(raw, fallback=None):
    m = list(re.finditer(r"<\|channel\|>\s*final\s*<\|message\|>", raw))
    seg = raw[m[-1].end():] if m else raw
    for seg_ in (seg, raw):
        starts = [i for i, ch in enumerate(seg_) if ch == "["]
        for i in reversed(starts):          # 右端の配列を優先
            try:
                o, _ = _dec.raw_decode(seg_[i:])
            except json.JSONDecodeError:
                continue
            if _valid(o):
                return o
        # 截断修復: 最後の '[' から、閉じた最後の '}' までで閉じる
        if starts:
            i = starts[-1]; c = seg_[i:]; j = c.rfind("}")
            if j > 0:
                try:
                    o = json.loads(c[:j + 1] + "]")
                    if _valid(o):
                        return o
                except json.JSONDecodeError:
                    pass
    return fallback(raw) if fallback else None
