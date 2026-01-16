from __future__ import annotations

import json
import unicodedata
from typing import Any


LONG_VOWEL = "\u30fc"


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"


def _is_hiragana(ch: str) -> bool:
    return "\u3040" <= ch <= "\u309f"


def _is_katakana(ch: str) -> bool:
    return "\u30a0" <= ch <= "\u30ff" or "\uff66" <= ch <= "\uff9d"


def _strip_punctuation(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "P")


def _strip_whitespace(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def _keep_by_lang(text: str, lang: str) -> str:
    if lang == "en":
        return "".join(ch for ch in text if "a" <= ch <= "z")
    if lang == "ja":
        return "".join(
            ch
            for ch in text
            if _is_cjk(ch)
            or _is_hiragana(ch)
            or _is_katakana(ch)
            or ch == LONG_VOWEL
        )
    # default zh
    return "".join(ch for ch in text if _is_cjk(ch))


def normalize_answer(text: str, lang: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = _strip_whitespace(text)
    text = _strip_punctuation(text)
    return _keep_by_lang(text, lang)


def is_correct(pred: str, truth: str, lang: str) -> bool:
    return normalize_answer(pred, lang) == normalize_answer(truth, lang)


_JSON_KEYS = ("text", "answer", "content", "response", "output")


def extract_prediction(raw: str) -> str:
    if raw is None:
        return ""
    raw = str(raw).strip()
    if not raw:
        return ""

    # Try JSON first (either object or list)
    try:
        parsed: Any = json.loads(raw)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        for key in _JSON_KEYS:
            if key in parsed and parsed[key] is not None:
                return str(parsed[key]).strip()
        return raw

    if isinstance(parsed, list) and parsed:
        for item in parsed:
            if isinstance(item, dict):
                for key in _JSON_KEYS:
                    if key in item and item[key] is not None:
                        return str(item[key]).strip()
        return raw

    # Fallback: strip enclosing quotes if the model returned quoted string
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1].strip()
    return raw


def summarize_confusion(pred: str, truth: str) -> str:
    pred = pred.strip()
    truth = truth.strip()
    if pred == truth:
        return "match"
    if not pred:
        return "empty_pred"
    if not truth:
        return "empty_truth"
    if normalize_answer(pred, "en") == normalize_answer(truth, "en"):
        return "normalized_match"
    return "mismatch"
