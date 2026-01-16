from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LANGS = {"en", "zh", "ja"}
VARIANT_LABELS = {
    1: "b",  # part2 hflip (right half horizontal) -> (b)
    2: "a",  # part2 vflip (right half vertical) -> (a)
    3: "d",  # part1 hflip (left half horizontal) -> (d)
    4: "c",  # part1 vflip (left half vertical) -> (c)
}
MODE_LABELS = {
    "tb": "a",   # horizontal split
    "lr": "b",   # vertical split
    "diag": "c", # diagonal split
}


@dataclass
class Sample:
    sample_id: str
    task: str
    lang: str
    image_path: Path
    ground_truth: str
    variant: str | None = None
    mode: str | None = None
    meta: dict[str, Any] | None = None


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"


def _is_hiragana(ch: str) -> bool:
    return "\u3040" <= ch <= "\u309f"


def _is_katakana(ch: str) -> bool:
    return "\u30a0" <= ch <= "\u30ff" or "\uff66" <= ch <= "\uff9d"


def _detect_lang(text: str) -> str | None:
    if not text:
        return None
    has_hira = any(_is_hiragana(ch) for ch in text)
    has_kata = any(_is_katakana(ch) for ch in text)
    has_cjk = any(_is_cjk(ch) for ch in text)
    if has_hira or has_kata:
        return "ja"
    if has_cjk:
        return "zh"
    if all("a" <= ch.lower() <= "z" for ch in text if ch.isascii()):
        return "en"
    return None


def _normalize_whitespace(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def _clean_ground_truth(text: str) -> str:
    return _normalize_whitespace(unicodedata.normalize("NFKC", text))


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _resolve_image_path(labels_dir: Path, filename: str) -> Path:
    return (labels_dir / filename).resolve()


def _path_relative_to_root(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root))
    except Exception:
        return str(path)


def _sample_from_record(
    record: dict[str, Any],
    labels_path: Path,
    dataset_root: Path,
    line_number: int,
) -> Sample | None:
    rule = record.get("rule")
    labels_dir = labels_path.parent

    task = None
    filename = None
    ground_truth = None
    lang = record.get("lang")
    variant = None
    mode = None
    meta: dict[str, Any] = {
        "labels_path": _path_relative_to_root(labels_path, dataset_root),
        "line_number": line_number,
    }

    if rule == "flip_overlay":
        task = "flip_overlay"
        filename = record.get("filename") or record.get("images", {}).get("fused")
        ground_truth = record.get("word") or record.get("text") or record.get("raw")
        variant_id = _safe_int(record.get("variant"))
        if variant_id:
            variant = VARIANT_LABELS.get(variant_id)
            meta["variant_id"] = variant_id
            meta["variant_label"] = variant
            meta["variant_name"] = record.get("variant_name")
    elif rule == "color_gradient_overlay":
        task = "gradient_overlay"
        filename = record.get("filename") or record.get("images", {}).get("fused")
        ground_truth = record.get("text") or record.get("raw_text")
    elif "pairs" in record and ("image" in record or "filename" in record):
        task = "cin_fusion"
        filename = record.get("image") or record.get("filename")
        ground_truth = record.get("raw_text") or record.get("text")
        mode = record.get("mode_name")
        if mode:
            meta["mode_label"] = MODE_LABELS.get(mode)
        else:
            modes = record.get("modes")
            if isinstance(modes, list):
                uniq = sorted({str(m) for m in modes if m})
                mode = "+".join(uniq) if uniq else None
        if not lang:
            lang = "zh"
    else:
        return None

    if not filename or not ground_truth:
        return None

    if not lang:
        lang = _detect_lang(str(ground_truth)) or "zh"

    if lang not in LANGS:
        lang = "zh"

    ground_truth = _clean_ground_truth(str(ground_truth))
    image_abs = _resolve_image_path(labels_dir, str(filename))

    sample = Sample(
        sample_id="",
        task=str(task),
        lang=str(lang),
        image_path=image_abs,
        ground_truth=ground_truth,
        variant=variant,
        mode=mode,
        meta=meta,
    )
    return sample


def iter_label_files(dataset_root: Path) -> Iterable[Path]:
    for path in sorted(dataset_root.rglob("labels.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        yield path


def build_manifest(dataset_root: Path, output_path: Path) -> dict[str, Any]:
    samples: list[Sample] = []
    counters: dict[tuple[str, str], int] = {}
    skipped = 0

    for labels_path in iter_label_files(dataset_root):
        try:
            lines = labels_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        for idx, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                skipped += 1
                continue
            sample = _sample_from_record(record, labels_path, dataset_root, idx)
            if not sample:
                skipped += 1
                continue

            key = (sample.task, sample.lang)
            counters[key] = counters.get(key, 0) + 1
            sample.sample_id = f"{sample.task}_{sample.lang}_{counters[key]:05d}"
            samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            record = {
                "id": sample.sample_id,
                "task": sample.task,
                "lang": sample.lang,
                "image_path": _path_relative_to_root(sample.image_path, dataset_root),
                "ground_truth": sample.ground_truth,
                "variant": sample.variant,
                "mode": sample.mode,
                "meta": sample.meta or {},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "total": len(samples),
        "skipped": skipped,
    }
    return summary


def load_manifest(manifest_path: Path, dataset_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    if not manifest_path.exists():
        return samples

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        image_path = Path(str(record.get("image_path", "")))
        if not image_path.is_absolute():
            image_path = (dataset_root / image_path).resolve()
        samples.append(
            Sample(
                sample_id=str(record.get("id", "")),
                task=str(record.get("task", "")),
                lang=str(record.get("lang", "")),
                image_path=image_path,
                ground_truth=str(record.get("ground_truth", "")),
                variant=record.get("variant"),
                mode=record.get("mode"),
                meta=record.get("meta") or {},
            )
        )
    return samples
