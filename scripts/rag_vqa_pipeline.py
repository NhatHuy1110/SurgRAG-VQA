"""
rag_vqa_pipeline.py - End-to-end RAG + VLM pipeline for surgical VQA.

Usage:
    export VLM_PROVIDER="local_hf"
    python scripts/rag_vqa_pipeline.py

Runs questions_v3.json against frames + retrieval_v3,
and saves structured results to results/spike_results_v3.json.
"""

import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FRAMES_DIR,
    HF_CACHE_DIR,
    HF_LOCAL_FILES_ONLY,
    HF_TOKEN,
    LOCAL_VLM_MAX_NEW_TOKENS,
    LOCAL_VLM_MODEL,
    OPENAI_API_KEY,
    QUESTIONS_FILE,
    RESULTS_FILE,
    RETRIEVAL_TOP_K,
    VLM_MAX_TOKENS,
    VLM_MODEL,
    VLM_PROVIDER,
    VLM_TEMPERATURE,
)
try:
    from scripts.retrieval import SurgicalRetriever
except ImportError:
    from retrieval import SurgicalRetriever


def encode_image_b64(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")


_LOCAL_VLM = None
_LOCAL_PROCESSOR = None
_LOCAL_DEVICE = None
_LOCAL_MODEL_KIND = "generic"

PROMPT_HINTS = {
    "recognition": "Focus on the main visible structure, tissue, or instrument interaction.",
    "workflow_phase": "Prefer broad workflow stages over overly specific claims from a single frame.",
    "anatomy_landmark": "Name the key landmark or safe orientation relationship only if it is visually defensible.",
    "safety_verification": "Do not claim the critical view of safety unless the image clearly supports it.",
    "risk_pitfall": "Prefer specific operative risks over vague warnings when they are actually supported.",
}

CONFIDENCE_RE = re.compile(
    r"confidence\s*[:=\-]?\s*\[?\s*(high|medium|low)\s*\]?",
    re.IGNORECASE,
)
ANSWER_PREFIX_RE = re.compile(r"\banswer\s*:\s*", re.IGNORECASE)
DEFER_PREFIX_RE = re.compile(r"^\s*defer\b\s*:?", re.IGNORECASE)
ANSWER_DEFER_RE = re.compile(r"\banswer\s*:\s*\[?\s*defer\b", re.IGNORECASE)
TEMPLATE_LEAK_PATTERNS = (
    "[your concise answer]",
    "[brief reason",
    "[high/medium/low]",
)
DEFER_CUES = (
    "should defer",
    "defer is recommended",
    "deferral is recommended",
    "cannot answer safely",
    "cannot determine safely",
    "not enough information",
    "insufficient visual evidence",
    "insufficient evidence",
    "image is too blurry",
    "image is too dark",
    "too ambiguous",
    "unable to identify confidently",
)
ASSISTANT_PREFIXES = (
    "assistant:",
    "assistant\n",
    "<|assistant|>",
    "[/inst]",
)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _truncate_text(text: str, max_chars: int) -> str:
    text = _collapse_ws(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _question_hint(question_type: Optional[str]) -> str:
    return PROMPT_HINTS.get(question_type or "", "")


def _max_prompt_evidence(question_type: Optional[str], compact: bool = False) -> int:
    base = 2 if question_type in {"recognition", "workflow_phase"} else 3
    return 1 if compact else base


def _clean_generation_text(text: str, prompt: str = "") -> str:
    cleaned = (text or "").replace("\x00", "").strip()
    for prefix in ASSISTANT_PREFIXES:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].lstrip()
    if prompt:
        prompt_norm = _collapse_ws(prompt)
        cleaned_norm = _collapse_ws(cleaned)
        if cleaned_norm.startswith(prompt_norm):
            cleaned = cleaned[len(prompt):].lstrip()
    return cleaned.strip()


def _detect_template_leak(text: str) -> bool:
    lower = (text or "").lower()
    return any(pat in lower for pat in TEMPLATE_LEAK_PATTERNS)


def _looks_like_garbage(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if re.fullmatch(r"(?:\[?0(?:\.0)?\]?[,\s.]*){6,}", stripped):
        return True
    if len(re.findall(r"[A-Za-z]", stripped)) == 0 and len(stripped) > 12:
        return True
    return False


def _raw_quality_flags(raw: str) -> tuple[str, list[str]]:
    text = _clean_generation_text(raw)
    flags = []
    if not text:
        flags.append("empty_raw")
        return text, flags
    if _detect_template_leak(text):
        flags.append("template_leak")
    if _looks_like_garbage(text):
        flags.append("garbage_output")
    if not (text.upper().startswith("ANSWER:") or text.upper().startswith("DEFER:")):
        flags.append("noncanonical_prefix")
    return text, flags


def _should_retry_output(raw: str) -> bool:
    _, flags = _raw_quality_flags(raw)
    return any(flag in flags for flag in ("empty_raw", "template_leak", "garbage_output"))


def _looks_like_defer(text: str) -> bool:
    lower = (text or "").lower()
    if DEFER_PREFIX_RE.match(text or ""):
        return True
    if ANSWER_DEFER_RE.search(text or ""):
        return True
    return any(cue in lower for cue in DEFER_CUES)


def _extract_confidence(text: str) -> str:
    match = CONFIDENCE_RE.search(text or "")
    return match.group(1).lower() if match else "unknown"


def _strip_confidence_tail(text: str) -> str:
    text = re.sub(
        r"\|\s*confidence\s*[:=\-]?\s*\[?\s*(?:high|medium|low)\s*\]?\s*$",
        "",
        text or "",
        flags=re.IGNORECASE,
    )
    return text.strip()


def _package_parse_result(
    raw_response: str,
    parsed_answer: str,
    is_defer: bool,
    confidence: str = "unknown",
    parse_flags: Optional[list[str]] = None,
) -> dict:
    return {
        "raw_response": raw_response,
        "is_defer": bool(is_defer),
        "confidence": confidence if not is_defer else "unknown",
        "parsed_answer": _collapse_ws(parsed_answer),
        "parse_flags": parse_flags or [],
    }


def build_system_prompt(
    retrieved_chunks: list[tuple[dict, float]],
    question_type: Optional[str] = None,
    compact: bool = False,
) -> tuple[str, list[dict]]:
    evidence_limit = _max_prompt_evidence(question_type, compact=compact)
    excerpt_chars = 240 if compact else 420
    selected_cards = []
    used_evidence_ids = set()
    pieces = []

    for chunk, _score in retrieved_chunks:
        card = chunk.get("evidence_card", {})
        evidence_id = chunk.get("evidence_chunk_id") or chunk.get("chunk_id")
        if evidence_id in used_evidence_ids:
            continue
        used_evidence_ids.add(evidence_id)

        meta_bits = []
        if card.get("doc_title"):
            meta_bits.append(card["doc_title"])
        elif chunk.get("doc_title"):
            meta_bits.append(chunk["doc_title"])
        if card.get("collection"):
            meta_bits.append(card["collection"])
        if card.get("section_title"):
            meta_bits.append(card["section_title"])

        evidence_text = chunk.get("evidence_raw_text") or chunk.get("text", "")
        evidence_text = _truncate_text(evidence_text, excerpt_chars)
        pieces.append(
            f"- Evidence {len(pieces) + 1}: {' | '.join(meta_bits)}\n"
            f"  {evidence_text}"
        )
        selected_cards.append(card or {"chunk_id": evidence_id})
        if len(pieces) >= evidence_limit:
            break

    context_block = (
        "\n".join(pieces)
        if pieces
        else "- No retrieved evidence available. If the image alone is not enough, defer."
    )
    qtype_hint = _question_hint(question_type)
    prompt = (
        "You are a cautious surgical AI assistant for laparoscopic cholecystectomy.\n\n"
        "Use the IMAGE as the primary source of truth. Use retrieved text only as supporting context.\n"
        "If the image is ambiguous, low quality, or the evidence is weak or conflicting, defer.\n\n"
        "Output rules:\n"
        "- Return exactly one line.\n"
        "- Do not use brackets.\n"
        "- Do not repeat the template.\n"
        "- If answering: ANSWER: <concise answer> | CONFIDENCE: high|medium|low\n"
        "- If unsafe or uncertain: DEFER: <brief reason>\n\n"
        "Examples:\n"
        "ANSWER: The frame most likely shows gallbladder retraction with a grasper. | CONFIDENCE: medium\n"
        "DEFER: The anatomy is too ambiguous to answer safely from this frame.\n\n"
    )
    if qtype_hint:
        prompt += f"Question-type hint: {qtype_hint}\n\n"
    prompt += "Retrieved evidence:\n" + context_block
    return prompt, selected_cards


def call_vlm(
    system_prompt: str,
    question: str,
    image_path: str | Path,
    mime: str = "image/jpeg",
    model: str = VLM_MODEL,
) -> str:
    if VLM_PROVIDER == "openai":
        return call_openai_vlm(system_prompt, question, image_path, mime, model)
    if VLM_PROVIDER == "local_hf":
        return call_local_hf_vlm(system_prompt, question, image_path)
    if VLM_PROVIDER == "mock_vlm":
        return (
            "DEFER: mock_vlm mode is enabled, so no vision-language model was called. "
            "Use this mode to validate retrieval and pipeline wiring only."
        )
    raise ValueError(f"Unsupported VLM_PROVIDER: {VLM_PROVIDER}")


def call_openai_vlm(
    system_prompt: str,
    question: str,
    image_path: str | Path,
    mime: str = "image/jpeg",
    model: str = VLM_MODEL,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    image_b64 = encode_image_b64(image_path)

    response = client.chat.completions.create(
        model=model,
        temperature=VLM_TEMPERATURE,
        max_tokens=VLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": question},
                ],
            },
        ],
    )
    return response.choices[0].message.content.strip()


def _load_local_hf_vlm():
    global _LOCAL_VLM, _LOCAL_PROCESSOR, _LOCAL_DEVICE, _LOCAL_MODEL_KIND
    if _LOCAL_VLM is not None and _LOCAL_PROCESSOR is not None:
        return _LOCAL_VLM, _LOCAL_PROCESSOR, _LOCAL_DEVICE

    import torch
    from transformers import AutoProcessor

    model_name = LOCAL_VLM_MODEL.lower()
    is_qwen25_vl = "qwen2.5" in model_name and "vl" in model_name
    is_qwen_vl = "qwen" in model_name and "vl" in model_name

    processor = AutoProcessor.from_pretrained(
        LOCAL_VLM_MODEL,
        cache_dir=HF_CACHE_DIR or None,
        local_files_only=HF_LOCAL_FILES_ONLY,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )

    if is_qwen25_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            LOCAL_VLM_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=HF_CACHE_DIR or None,
            local_files_only=HF_LOCAL_FILES_ONLY,
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )
        device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
        _LOCAL_MODEL_KIND = "qwen_vl"
    elif is_qwen_vl:
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            LOCAL_VLM_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=HF_CACHE_DIR or None,
            local_files_only=HF_LOCAL_FILES_ONLY,
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )
        device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
        _LOCAL_MODEL_KIND = "qwen_vl"
    else:
        from transformers import AutoModelForImageTextToText

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            LOCAL_VLM_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=HF_CACHE_DIR or None,
            local_files_only=HF_LOCAL_FILES_ONLY,
            token=HF_TOKEN or None,
        )
        model.to(device)
        _LOCAL_MODEL_KIND = "generic"

    model.eval()

    _LOCAL_VLM = model
    _LOCAL_PROCESSOR = processor
    _LOCAL_DEVICE = str(device)
    return _LOCAL_VLM, _LOCAL_PROCESSOR, _LOCAL_DEVICE


def call_local_hf_vlm(
    system_prompt: str,
    question: str,
    image_path: str | Path,
) -> str:
    model, processor, device = _load_local_hf_vlm()
    prompt = (
        f"{system_prompt}\n\n"
        f"Question: {question}\n"
        "Return only the final one-line answer."
    )

    if _LOCAL_MODEL_KIND == "qwen_vl":
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=LOCAL_VLM_MAX_NEW_TOKENS,
                do_sample=False,
            )
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated = generated[:, input_ids.shape[-1]:]
        text = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return _clean_generation_text(text, prompt=prompt)

    import torch
    from PIL import Image

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
        rendered_prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=rendered_prompt,
            images=[image],
            return_tensors="pt",
        )
    except Exception:
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

    inputs = {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=LOCAL_VLM_MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        generated = generated[:, input_ids.shape[-1]:]

    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return _clean_generation_text(text, prompt=prompt)


def parse_response(raw: str) -> dict:
    """Extract structured fields from the VLM's raw text response."""
    text, flags = _raw_quality_flags(raw)
    confidence = _extract_confidence(text)

    if not text:
        return _package_parse_result(
            raw_response="",
            parsed_answer="The model returned an empty response, so the safe action is to defer.",
            is_defer=True,
            parse_flags=flags,
        )

    if "template_leak" in flags:
        return _package_parse_result(
            raw_response=text,
            parsed_answer="The model repeated the response template instead of giving a reliable answer.",
            is_defer=True,
            parse_flags=flags,
        )

    if "garbage_output" in flags:
        return _package_parse_result(
            raw_response=text,
            parsed_answer="The model output was malformed and could not be trusted safely.",
            is_defer=True,
            parse_flags=flags,
        )

    is_defer = _looks_like_defer(text)
    answer_text = text

    if is_defer:
        defer_match = re.search(r"\bdefer\b\s*:?\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        if defer_match and defer_match.group(1).strip():
            answer_text = defer_match.group(1).strip()
        else:
            answer_text = text
        return _package_parse_result(
            raw_response=text,
            parsed_answer=_strip_confidence_tail(answer_text),
            is_defer=True,
            parse_flags=flags,
        )

    answer_match = ANSWER_PREFIX_RE.search(text)
    if answer_match:
        answer_text = text[answer_match.end():].strip()
    answer_text = _strip_confidence_tail(answer_text)
    if not answer_text:
        flags.append("blank_answer_after_parse")
        return _package_parse_result(
            raw_response=text,
            parsed_answer="The model did not provide a usable answer after parsing, so the safe action is to defer.",
            is_defer=True,
            parse_flags=flags,
        )

    return _package_parse_result(
        raw_response=text,
        parsed_answer=answer_text,
        is_defer=False,
        confidence=confidence,
        parse_flags=flags,
    )


def run_single(
    frame_path: Path,
    question: str,
    retriever: SurgicalRetriever,
    top_k: int = RETRIEVAL_TOP_K,
    question_type: Optional[str] = None,
    classes_detected: Optional[dict] = None,
) -> dict:
    """Run retrieval -> prompt -> VLM -> parse for one question."""
    retrieved = retriever.retrieve_hybrid(
        question,
        top_k=top_k,
        question_type=question_type,
        classes_detected=classes_detected,
    )

    system_prompt, prompt_cards = build_system_prompt(
        retrieved,
        question_type=question_type,
        compact=False,
    )
    mime = detect_mime(frame_path)
    raw = call_vlm(system_prompt, question, frame_path, mime)
    prompt_variant = "full"

    if VLM_PROVIDER == "local_hf" and _should_retry_output(raw):
        compact_prompt, compact_cards = build_system_prompt(
            retrieved,
            question_type=question_type,
            compact=True,
        )
        retry_raw = call_vlm(compact_prompt, question, frame_path, mime)
        retry_text, retry_flags = _raw_quality_flags(retry_raw)
        raw_text, raw_flags = _raw_quality_flags(raw)
        if (len(retry_flags), -len(retry_text)) <= (len(raw_flags), -len(raw_text)):
            raw = retry_raw
            prompt_cards = compact_cards
            prompt_variant = "compact_retry"

    parsed = parse_response(raw)

    return {
        "frame": str(frame_path.name),
        "question": question,
        "retrieved_chunks": [c["chunk_id"] for c, _ in retrieved],
        "retrieved_matched_chunks": [c.get("matched_chunk_id", c["chunk_id"]) for c, _ in retrieved],
        "retrieved_evidence_chunks": [c.get("evidence_chunk_id", c["chunk_id"]) for c, _ in retrieved],
        "retrieved_scores": [round(s, 3) for _, s in retrieved],
        "retrieved_previews": [c["text"][:200] for c, _ in retrieved],
        "retrieved_evidence_cards": [c.get("evidence_card", {}) for c, _ in retrieved],
        "prompt_variant": prompt_variant,
        "prompt_evidence_count": len(prompt_cards),
        "local_model_kind": _LOCAL_MODEL_KIND if VLM_PROVIDER == "local_hf" else VLM_PROVIDER,
        **parsed,
    }


def run_all(
    retriever: SurgicalRetriever,
    questions_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
):
    questions_path = questions_path or QUESTIONS_FILE
    output_path = output_path or RESULTS_FILE

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    print(f"\n{'=' * 60}")
    print(f"Running RAG-VQA pipeline on {len(questions)} questions")
    print(f"VLM: {VLM_PROVIDER}:{VLM_MODEL}  |  Retrieval top-k: {RETRIEVAL_TOP_K}")
    print(f"{'=' * 60}\n")

    for q in questions:
        qid = q["qid"]
        frame_path = FRAMES_DIR / q["frame"]

        if not frame_path.exists():
            print(f"  [WARN] {qid}: Frame not found -> {frame_path} (skipping)")
            results.append(
                {
                    "qid": qid,
                    "frame": q["frame"],
                    "question": q["question"],
                    "error": f"Frame not found: {frame_path}",
                    "raw_response": "",
                    "is_defer": True,
                    "confidence": "unknown",
                    "parsed_answer": "",
                    "parse_flags": ["frame_missing"],
                    "prompt_variant": "not_run",
                    "prompt_evidence_count": 0,
                    "gold_answer": q["gold_answer"],
                    "should_defer": q["should_defer"],
                    "question_type": q["question_type"],
                    "difficulty": q["difficulty"],
                }
            )
            continue

        print(f"  {qid}: {q['question'][:55]}...", end="", flush=True)
        t0 = time.time()

        try:
            result = run_single(
                frame_path,
                q["question"],
                retriever,
                question_type=q.get("question_type"),
                classes_detected=q.get("classes_detected"),
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            result = {
                "frame": str(frame_path.name),
                "question": q["question"],
                "raw_response": "",
                "is_defer": True,
                "confidence": "unknown",
                "parsed_answer": "",
                "retrieved_chunks": [],
                "retrieved_matched_chunks": [],
                "retrieved_evidence_chunks": [],
                "retrieved_scores": [],
                "retrieved_previews": [],
                "retrieved_evidence_cards": [],
                "parse_flags": ["generation_error"],
                "prompt_variant": "error",
                "prompt_evidence_count": 0,
                "error": str(e),
            }

        elapsed = time.time() - t0

        result["qid"] = qid
        result["gold_answer"] = q["gold_answer"]
        result["should_defer"] = q["should_defer"]
        result["question_type"] = q["question_type"]
        result["difficulty"] = q["difficulty"]
        result["latency_s"] = round(elapsed, 2)
        results.append(result)

        if "error" in result:
            status = "ERROR"
        else:
            status = "DEFER" if result.get("is_defer") else f"ANS ({result.get('confidence', '?')})"
        print(f"  -> {status}  [{elapsed:.1f}s]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(results)} results -> {output_path}")
    answered = sum(1 for r in results if "error" not in r and not r.get("is_defer"))
    deferred = sum(1 for r in results if r.get("is_defer"))
    errors = sum(1 for r in results if "error" in r)
    print(f"  Answered: {answered}  |  Deferred: {deferred}  |  Errors: {errors}")
    print(f"{'=' * 60}")

    return results


def run_mock(retriever: SurgicalRetriever):
    """
    Run retrieval only - useful when you do not want to call a VLM yet.
    Shows what chunks would be retrieved for each question.
    """
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n{'=' * 60}")
    print("MOCK MODE - retrieval only, no VLM call")
    print(f"{'=' * 60}\n")

    mock_results = []
    for q in questions:
        retrieved = retriever.retrieve_hybrid(
            q["question"],
            top_k=RETRIEVAL_TOP_K,
            question_type=q.get("question_type"),
            classes_detected=q.get("classes_detected"),
        )
        print(f"\n{q['qid']}: {q['question']}")
        for rank, (chunk, score) in enumerate(retrieved, 1):
            print(f"  #{rank} [{score:.3f}] {chunk['text'][:120]}...")

        mock_results.append(
            {
                "qid": q["qid"],
                "question": q["question"],
                "retrieved_chunks": [c["chunk_id"] for c, _ in retrieved],
                "retrieved_matched_chunks": [c.get("matched_chunk_id", c["chunk_id"]) for c, _ in retrieved],
                "retrieved_evidence_chunks": [c.get("evidence_chunk_id", c["chunk_id"]) for c, _ in retrieved],
                "retrieved_scores": [round(s, 3) for _, s in retrieved],
                "gold_answer": q["gold_answer"],
                "should_defer": q["should_defer"],
                "question_type": q["question_type"],
                "mode": "mock_retrieval_only",
            }
        )

    mock_path = RESULTS_FILE.parent / "mock_retrieval_results.json"
    with open(mock_path, "w", encoding="utf-8") as f:
        json.dump(mock_results, f, indent=2, ensure_ascii=False)
    print(f"\nMock results saved -> {mock_path}")


def main():
    retriever = SurgicalRetriever()

    if VLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        print("\n[WARN] OPENAI_API_KEY not set - running in MOCK mode (retrieval only)")
        run_mock(retriever)
        return

    if VLM_PROVIDER == "local_hf":
        print(f"[Preflight] Loading local VLM: {LOCAL_VLM_MODEL}")
        _load_local_hf_vlm()

    run_all(retriever)


if __name__ == "__main__":
    main()
