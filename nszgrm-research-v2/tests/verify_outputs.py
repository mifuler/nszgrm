#!/usr/bin/env python3
"""Validate model-only training outputs, task schemas, fusion and size budget."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

EXPECTED_WRITING = {"zh_native", "ja_kanji_proxy", "foreign_phonetic_proxy"}
EXPECTED_LANGUAGE = {"zh", "en", "fr", "de", "es", "ru", "ja", "ko"}
EXPECTED_MODELS = {"linear", "complement_nb", "cnn", "bigru"}
BYTE_CAP = 200_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--allow-demo", action="store_true")
    ap.add_argument("--require-language", action="store_true")
    args = ap.parse_args()
    out = args.output
    for name in ("models_fp32.json", "REPORT.json", "REPORT.md", "train_manifest.json"):
        if not (out / name).is_file() or (out / name).stat().st_size == 0:
            raise SystemExit(f"missing or empty output: {name}")
    htmls = list(out.rglob("*.html"))
    if htmls:
        raise SystemExit(f"training output must not contain HTML: {htmls[:3]}")
    model_path = out / "models_fp32.json"
    size = model_path.stat().st_size
    if size > BYTE_CAP:
        raise SystemExit(f"FP32 model file exceeds 200 MB: {size}")
    bundle = json.loads(model_path.read_text(encoding="utf8"))
    report = json.loads((out / "REPORT.json").read_text(encoding="utf8"))
    if bundle.get('format_version') != 3 or report.get('trainer_version') != '3.0.0':
        raise SystemExit('unexpected model bundle / trainer version')
    if not args.allow_demo and report.get("training_mode") != "PUBLIC_WEAK_SUPERVISION":
        raise SystemExit("formal run must be PUBLIC_WEAK_SUPERVISION, not demo")
    tasks = bundle.get("models", {})
    if "writing" not in tasks:
        raise SystemExit("missing writing model")
    if set(tasks["writing"].get("classes", [])) != EXPECTED_WRITING:
        raise SystemExit(f"writing classes mismatch: {tasks['writing'].get('classes')}")
    for task_name, expected in (("writing", EXPECTED_WRITING), ("language", EXPECTED_LANGUAGE)):
        if task_name not in tasks:
            if task_name == "language" and not args.require_language:
                continue
            raise SystemExit(f"missing task model: {task_name}")
        task = tasks[task_name]
        if set(task.get("classes", [])) != expected:
            raise SystemExit(f"{task_name} classes mismatch: {task.get('classes')}")
        models = task.get("models", {})
        if set(models) != EXPECTED_MODELS:
            raise SystemExit(f"{task_name} models mismatch: {sorted(models)}")
        for model_name, model in models.items():
            if model.get("classes") != task["classes"]:
                raise SystemExit(f"{task_name}/{model_name} class order mismatch")
            if not isinstance(model.get("temperature"), (int, float)) or model["temperature"] <= 0:
                raise SystemExit(f"{task_name}/{model_name} has invalid calibration temperature")
        weights = task.get("fusion_weights", {})
        if (set(weights) != EXPECTED_MODELS or
                any(not (0.0 <= float(v) <= 1.0) for v in weights.values()) or
                abs(sum(weights.values()) - 1.0) > 1e-6):
            raise SystemExit(f"{task_name} fusion weights invalid: {weights}")
        metrics = report.get("task_reports", {}).get(task_name, {}).get("weak_test_metrics", {})
        if set(metrics) != EXPECTED_MODELS | {"fusion"}:
            raise SystemExit(f"{task_name} test metrics incomplete: {sorted(metrics)}")
    if args.require_language and set(tasks) != {"writing", "language"}:
        raise SystemExit(f"required two-task bundle missing language: {sorted(tasks)}")
    print(f"[PASS] model-only outputs; FP32={size:,}/{BYTE_CAP:,} bytes")
    print("[PASS] writing task: 3 classes, 4 models + fusion")
    if "language" in tasks:
        print("[PASS] language task: 8 classes including zh, 4 models + fusion")
    else:
        print("[WARN] language model unavailable; report records the reason")
    print("[PASS] no HTML generated")


if __name__ == "__main__":
    main()
