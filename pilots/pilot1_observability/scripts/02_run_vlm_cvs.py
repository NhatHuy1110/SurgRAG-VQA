from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


SYSTEM_INSTRUCTION = """You are assessing a single laparoscopic cholecystectomy frame for Critical View of Safety (CVS).
Answer only from visible evidence in the image.
If the required anatomy is not clearly visible, if the view is incomplete, or if you would need temporal context, answer "uncertain".
Do not infer anatomy from surgical context. Do not guess.
"""


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def enabled_models(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in cfg["vqa"]["models"] if m.get("enabled", False)]


def get_model_cfg(cfg: dict[str, Any], model_id: str) -> dict[str, Any]:
    for model in cfg["vqa"]["models"]:
        if model["id"] == model_id:
            return model
    raise ValueError(f"Unknown model_id: {model_id}")


def build_prompt(question: str) -> str:
    return f"""{SYSTEM_INSTRUCTION}

Official CVS criteria:
1. Two and only two structures enter the gallbladder.
2. The hepatocystic triangle is cleared of fat and fibrous tissue.
3. The lower gallbladder is separated from the liver bed, exposing the cystic plate.

Question:
{question}

Return strict JSON only:
{{
  "answer": "yes" | "no" | "uncertain",
  "evidence": "one short sentence explaining the visible evidence or missing evidence"
}}
"""


def dtype_from_config(value: str):
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    return torch.float16


def load_hf_model(model_cfg: dict[str, Any], cfg: dict[str, Any]):
    model_name = model_cfg["model_name"]
    adapter = model_cfg.get("adapter", "generic_hf")
    trust_remote_code = bool(cfg["vqa"].get("trust_remote_code", True))
    dtype = dtype_from_config(str(cfg["vqa"].get("torch_dtype", "float16")))

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    lower = model_name.lower()
    if adapter == "qwen_vl" or ("qwen" in lower and "vl" in lower):
        if "qwen2.5" in lower:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )

    model.eval()
    return model, processor


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_qwen_vl(model, processor, image_path: str, prompt: str, max_new_tokens: int, do_sample: bool) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(image_path).resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model_device(model))

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def run_generic_hf(model, processor, image_path: str, prompt: str, max_new_tokens: int, do_sample: bool) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=rendered, images=[image], return_tensors="pt")
    except Exception:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = inputs.to(model_device(model))

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    if hasattr(inputs, "input_ids"):
        generated_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def parse_answer(raw: str) -> dict[str, Any]:
    text = (raw or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(text)
        ans = str(obj.get("answer", "")).lower().strip()
        if ans not in {"yes", "no", "uncertain"}:
            ans = "unknown"
        return {
            "parsed_answer": ans,
            "evidence": str(obj.get("evidence", "")).strip(),
            "parse_ok": True,
        }
    except Exception:
        pass

    lower = text.lower()
    if re.search(r"\buncertain\b|\binsufficient\b|\bnot visible\b|\bcannot determine\b|\bunclear\b", lower):
        ans = "uncertain"
    elif re.search(r"\bno\b", lower):
        ans = "no"
    elif re.search(r"\byes\b", lower):
        ans = "yes"
    else:
        ans = "unknown"

    return {
        "parsed_answer": ans,
        "evidence": text[:500],
        "parse_ok": False,
    }


def existing_keys(path: Path) -> set[tuple[str, str]]:
    keys = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                keys.add((str(obj["sample_id"]), str(obj["criterion"])))
            except Exception:
                continue
    return keys


def run_model(config_path: Path, model_id: str, limit: int | None = None) -> None:
    cfg = load_config(config_path)
    root = Path(config_path).resolve().parents[2]
    output_dir = root / cfg["paths"]["output_dir"]
    manifest = pd.read_csv(output_dir / "pilot_manifest.csv")
    if limit is not None and limit > 0:
        manifest = manifest.head(limit).copy()

    model_cfg = get_model_cfg(cfg, model_id)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{model_id}.jsonl"
    done = existing_keys(pred_path)

    print(f"Loading model {model_id}: {model_cfg['model_name']}")
    model, processor = load_hf_model(model_cfg, cfg)

    max_new_tokens = int(cfg["vqa"]["max_new_tokens"])
    do_sample = bool(cfg["vqa"].get("do_sample", False))
    adapter = model_cfg.get("adapter", "generic_hf")

    with pred_path.open("a", encoding="utf-8") as f:
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"VQA {model_id}"):
            sample_id = str(row["record_id"])
            image_path = str(row["frame_path"])
            for criterion, criterion_cfg in cfg["cvs_criteria"].items():
                key = (sample_id, criterion)
                if key in done:
                    continue
                prompt = build_prompt(criterion_cfg["question"])

                try:
                    if adapter == "qwen_vl":
                        raw = run_qwen_vl(model, processor, image_path, prompt, max_new_tokens, do_sample)
                    else:
                        raw = run_generic_hf(model, processor, image_path, prompt, max_new_tokens, do_sample)
                    parsed = parse_answer(raw)
                    record = {
                        "model_id": model_id,
                        "model_name": model_cfg["model_name"],
                        "sample_id": sample_id,
                        "pilot_row_index": int(row["pilot_row_index"]),
                        "frame_path": image_path,
                        "criterion": criterion,
                        "question": criterion_cfg["question"],
                        "raw_output": raw,
                        **parsed,
                    }
                except Exception as exc:
                    record = {
                        "model_id": model_id,
                        "model_name": model_cfg["model_name"],
                        "sample_id": sample_id,
                        "pilot_row_index": int(row["pilot_row_index"]),
                        "frame_path": image_path,
                        "criterion": criterion,
                        "question": criterion_cfg["question"],
                        "raw_output": "",
                        "parsed_answer": "error",
                        "evidence": str(exc),
                        "parse_ok": False,
                    }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                done.add(key)

    print(f"Saved predictions: {pred_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("pilots/pilot1_observability/config.yaml"))
    parser.add_argument("--model-id", default="", help="Run one model id. If omitted, runs enabled models.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_ids = [args.model_id] if args.model_id else [m["id"] for m in enabled_models(cfg)]
    for model_id in model_ids:
        run_model(args.config, model_id, limit=args.limit or None)


if __name__ == "__main__":
    main()

