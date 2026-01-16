from __future__ import annotations

import base64
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from dataset import build_manifest, load_manifest, Sample
from messenger import Messenger
from scoring import extract_prediction, is_correct, normalize_answer


logger = logging.getLogger("vision_benchmark")
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_PROMPT = (
    "Read the text in this image and reply with only the text. "
    "Do not add explanations or extra punctuation."
)
DEFAULT_PROMPT_VERSION = "basic"
DEFAULT_STATUS_EVERY = 10
DEFAULT_MAX_ERRORS = 50

PROMPTS: dict[str, dict[str, dict[str, str]]] = {
    "flip_overlay": {
        "basic": {
            "en": "What text do you see in this image? Answer with just the text, no explanation or thinking.",
            "ja": "What text do you see in this image? Answer with just the text, no explanation or thinking.",
            "zh": "What text do you see in this image? Answer with just the text, no explanation or thinking.",
        },
        "detailed": {
            "en": (
                "This is an English word consisting of 8 letters, one part of which is flipped "
                "and fused together. What word is it? Answer with just the text, no explanation or thinking."
            ),
            "ja": (
                "This is a Japanese word composed of four kana characters, one part of which is flipped "
                "and fused together. What word is it? Answer with just the text, no explanation or thinking."
            ),
            "zh": (
                "This is a Chinese idiom consisting of four characters, one part of which is flipped "
                "and fused together. What idiom is it? Answer with just the text, no explanation or thinking."
            ),
        },
    },
    "gradient_overlay": {
        "basic": {
            "en": "What text do you see in this image? Answer with just the text, no explanation or thinking.",
            "ja": "What text do you see in this image? Answer with just the text, no explanation or thinking.",
            "zh": "What text do you see in this image? Answer with just the text, no explanation or thinking.",
        },
        "detailed": {
            "en": (
                "This is an English word consisting of 8 letters, one part of which is colored "
                "and fused together. What word is it? Answer with just the text, no explanation or thinking."
            ),
            "ja": (
                "This is a Japanese word composed of four kana characters, one part of which is colored "
                "and fused together. What word is it? Answer with just the text, no explanation or thinking."
            ),
            "zh": (
                "This is a Chinese idiom consisting of four characters, one part of which is colored "
                "and fused together. What idiom is it? Answer with just the text, no explanation or thinking."
            ),
        },
    },
    "cin_fusion": {
        "basic": {
            "zh": (
                "Identify the Chinese characters in the image and return only the recognized text, "
                "without any explanation."
            )
        },
        "detailed": {
            "zh": (
                "This image contains fused Chinese characters, where each character is composed of parts "
                "from two different characters. Carefully identify the original characters and return "
                "only the recognized text."
            )
        },
        "contextual": {
            "zh": (
                "This image shows an artistic rendering of a Chinese idiom, where each character is formed "
                "by fusing parts of two characters. Identify the complete four-character idiom."
            )
        },
    },
}


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


class Agent:
    required_roles: list[str] = ["agent"]
    required_config_keys: list[str] = []

    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Add additional request validation here

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        dataset_root = _resolve_dataset_root(request.config)
        manifest_path = _resolve_manifest_path(request.config, dataset_root)
        rebuild_manifest = bool(request.config.get("rebuild_manifest", False))
        build_if_missing = bool(request.config.get("build_manifest_if_missing", True))

        manifest_summary = None
        if rebuild_manifest or (build_if_missing and not manifest_path.exists()):
            manifest_summary = build_manifest(dataset_root, manifest_path)

        samples = load_manifest(manifest_path, dataset_root)
        if not samples:
            await updater.reject(
                new_agent_text_message(
                    f"No samples found. manifest={manifest_path} dataset_root={dataset_root}"
                )
            )
            return

        samples = _apply_filters(samples, request.config)
        if not samples:
            await updater.reject(new_agent_text_message("No samples after filtering."))
            return

        max_samples = _get_int(request.config.get("limit"))
        shuffle = bool(request.config.get("shuffle", False))
        seed = _get_int(request.config.get("seed"), default=42)
        if max_samples is not None and max_samples > 0:
            samples = _take_samples(samples, max_samples, shuffle=shuffle, seed=seed)

        agent_url = str(request.participants["agent"])
        model_name = str(request.config.get("model", DEFAULT_MODEL))
        prompt_default = str(request.config.get("prompt", DEFAULT_PROMPT))
        prompt_version = str(request.config.get("prompt_version", DEFAULT_PROMPT_VERSION))
        prompt_by_lang = request.config.get("prompt_by_lang", {})
        prompt_by_task = request.config.get("prompt_by_task", {})
        prompt_by_task_lang = request.config.get("prompt_by_task_lang", {})
        status_every = _get_int(request.config.get("status_every"), DEFAULT_STATUS_EVERY)
        max_errors = _get_int(request.config.get("max_error_samples"), DEFAULT_MAX_ERRORS)

        log_path = _prepare_log_path(dataset_root, request.config)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting evaluation of {len(samples)} samples. manifest={manifest_path}"
            ),
        )

        totals = _new_bucket()
        by_task: dict[str, dict[str, int]] = {}
        by_lang: dict[str, dict[str, int]] = {}
        by_task_lang: dict[str, dict[str, int]] = {}
        by_variant: dict[str, dict[str, int]] = {}
        by_mode: dict[str, dict[str, int]] = {}
        error_samples: list[dict[str, Any]] = []

        start_time = time.time()
        try:
            with log_path.open("w", encoding="utf-8") as log_f:
                for idx, sample in enumerate(samples, start=1):
                    if status_every and idx % status_every == 0:
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(f"Processed {idx}/{len(samples)} samples"),
                        )

                    prompt = _select_prompt(
                        task=sample.task,
                        lang=sample.lang,
                        version=prompt_version,
                        prompt_default=prompt_default,
                        prompt_by_lang=prompt_by_lang,
                        prompt_by_task=prompt_by_task,
                        prompt_by_task_lang=prompt_by_task_lang,
                    )
                    response_text = ""
                    prediction = ""
                    error_msg = None
                    try:
                        request_payload = _build_gemini_request(
                            sample, prompt=prompt, model=model_name
                        )
                        response_text = await self.messenger.talk_to_agent(
                            message=json.dumps(request_payload, ensure_ascii=False),
                            url=agent_url,
                        )
                        prediction = extract_prediction(response_text)
                    except Exception as e:
                        error_msg = str(e)

                    ok = is_correct(prediction, sample.ground_truth, sample.lang)

                    _update_bucket(totals, ok)
                    _update_bucket(by_task.setdefault(sample.task, _new_bucket()), ok)
                    _update_bucket(by_lang.setdefault(sample.lang, _new_bucket()), ok)
                    tl_key = f"{sample.task}:{sample.lang}"
                    _update_bucket(by_task_lang.setdefault(tl_key, _new_bucket()), ok)

                    if sample.variant:
                        _update_bucket(by_variant.setdefault(sample.variant, _new_bucket()), ok)
                    if sample.mode:
                        _update_bucket(by_mode.setdefault(sample.mode, _new_bucket()), ok)

                    record = {
                        "id": sample.sample_id,
                        "task": sample.task,
                        "lang": sample.lang,
                        "variant": sample.variant,
                        "mode": sample.mode,
                        "image_path": str(sample.image_path),
                        "ground_truth": sample.ground_truth,
                        "prediction": prediction,
                        "normalized": {
                            "pred": normalize_answer(prediction, sample.lang),
                            "truth": normalize_answer(sample.ground_truth, sample.lang),
                        },
                        "correct": ok,
                        "response_raw": response_text,
                    }
                    if error_msg:
                        record["error"] = error_msg
                    log_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if not ok and len(error_samples) < max_errors:
                        error_samples.append(record)
        finally:
            self.messenger.reset()

        time_used = time.time() - start_time
        result_data = {
            "total": totals["total"],
            "correct": totals["correct"],
            "accuracy": _accuracy(totals),
            "by_task": _bucket_summary(by_task),
            "by_lang": _bucket_summary(by_lang),
            "by_task_lang": _bucket_summary(by_task_lang),
            "by_variant": _bucket_summary(by_variant),
            "by_mode": _bucket_summary(by_mode),
            "log_path": str(log_path),
            "manifest_path": str(manifest_path),
            "manifest_summary": manifest_summary,
            "time_used_sec": round(time_used, 3),
            "error_samples": error_samples,
        }

        summary_text = _build_summary_text(result_data)
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=result_data)),
            ],
            name="Result",
        )


def _resolve_dataset_root(config: dict[str, Any]) -> Path:
    root = config.get("dataset_root")
    if root:
        return Path(str(root)).resolve()
    return Path(__file__).resolve().parents[1] / "datasets"


def _resolve_manifest_path(config: dict[str, Any], dataset_root: Path) -> Path:
    path = config.get("manifest_path")
    if path:
        return Path(str(path)).resolve()
    return dataset_root / "manifest.jsonl"


def _get_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _parse_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
    else:
        items = [x.strip() for x in str(value).split(",") if x.strip()]
    return set(items) if items else None


def _apply_filters(samples: list[Sample], config: dict[str, Any]) -> list[Sample]:
    tasks = _parse_set(config.get("tasks"))
    langs = _parse_set(config.get("langs"))
    if tasks:
        samples = [s for s in samples if s.task in tasks]
    if langs:
        samples = [s for s in samples if s.lang in langs]
    return samples


def _take_samples(samples: list[Sample], limit: int, shuffle: bool, seed: int) -> list[Sample]:
    if limit <= 0:
        return []
    samples = samples[:]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    return samples[:limit]


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def _read_image_b64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _select_prompt(
    *,
    task: str,
    lang: str,
    version: str,
    prompt_default: str,
    prompt_by_lang: Any,
    prompt_by_task: Any,
    prompt_by_task_lang: Any,
) -> str:
    if isinstance(prompt_by_task_lang, dict):
        task_map = prompt_by_task_lang.get(task)
        if isinstance(task_map, dict):
            lang_prompt = task_map.get(lang)
            if isinstance(lang_prompt, str) and lang_prompt.strip():
                return lang_prompt.strip()

    if isinstance(prompt_by_task, dict):
        task_prompt = prompt_by_task.get(task)
        if isinstance(task_prompt, str) and task_prompt.strip():
            return task_prompt.strip()

    if isinstance(prompt_by_lang, dict):
        lang_prompt = prompt_by_lang.get(lang)
        if isinstance(lang_prompt, str) and lang_prompt.strip():
            return lang_prompt.strip()

    task_map = PROMPTS.get(task, {})
    if isinstance(task_map, dict):
        version_map = task_map.get(version) or task_map.get("basic") or {}
        if isinstance(version_map, dict):
            lang_prompt = version_map.get(lang)
            if isinstance(lang_prompt, str) and lang_prompt.strip():
                return lang_prompt.strip()
            if len(version_map) == 1:
                return next(iter(version_map.values()))

    return prompt_default


def _build_gemini_request(sample: Sample, prompt: str, model: str) -> dict[str, Any]:
    image_b64 = _read_image_b64(sample.image_path)
    mime_type = _guess_mime_type(sample.image_path)
    return {
        "id": sample.sample_id,
        "task": sample.task,
        "lang": sample.lang,
        "variant": sample.variant,
        "mode": sample.mode,
        "gemini": {
            "model": model,
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": image_b64,
                            }
                        },
                    ],
                }
            ],
        },
    }


def _prepare_log_path(dataset_root: Path, config: dict[str, Any]) -> Path:
    logs_dir = dataset_root.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = str(config.get("log_suffix", ""))
    if suffix:
        suffix = f"_{suffix}"
    return logs_dir / f"responses_{timestamp}{suffix}.jsonl"


def _new_bucket() -> dict[str, int]:
    return {"total": 0, "correct": 0}


def _update_bucket(bucket: dict[str, int], ok: bool) -> None:
    bucket["total"] += 1
    if ok:
        bucket["correct"] += 1


def _accuracy(bucket: dict[str, int]) -> float:
    if bucket["total"] <= 0:
        return 0.0
    return bucket["correct"] / bucket["total"]


def _bucket_summary(buckets: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, bucket in buckets.items():
        out[key] = {
            "total": bucket["total"],
            "correct": bucket["correct"],
            "accuracy": _accuracy(bucket),
        }
    return out


def _build_summary_text(result: dict[str, Any]) -> str:
    total = result.get("total", 0)
    correct = result.get("correct", 0)
    accuracy = result.get("accuracy", 0.0)
    time_used = result.get("time_used_sec", 0.0)
    return (
        "Benchmark Results\n"
        f"Samples: {total}\n"
        f"Accuracy: {accuracy:.2%} ({correct}/{total})\n"
        f"Time: {time_used:.2f}s\n"
        f"Log: {result.get('log_path')}"
    )
