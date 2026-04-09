"""
rag_vqa_pipeline.py — End-to-end RAG + VLM pipeline for surgical VQA.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/rag_vqa_pipeline.py

Runs all 10 questions from questions_v1.json against frames + retrieval,
and saves structured results to results/spike_results_v1.json.
"""

import json
import base64
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    OPENAI_API_KEY, VLM_PROVIDER, VLM_MODEL, VLM_MAX_TOKENS, VLM_TEMPERATURE,
    LOCAL_VLM_MODEL, LOCAL_VLM_MAX_NEW_TOKENS,
    HF_TOKEN,
    HF_CACHE_DIR, HF_LOCAL_FILES_ONLY,
    FRAMES_DIR, QUESTIONS_FILE, RESULTS_FILE, RETRIEVAL_TOP_K,
)
from retrieval import SurgicalRetriever


# ─── Image encoding ─────────────────────────────────────────────────

def encode_image_b64(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
    }.get(suffix, "image/jpeg")


_LOCAL_VLM = None
_LOCAL_PROCESSOR = None
_LOCAL_DEVICE = None


# ─── Prompt construction ────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a surgical AI assistant for laparoscopic cholecystectomy.

YOUR TASK:
Analyze the provided surgical frame image together with the retrieved surgical 
knowledge below. Answer the surgeon's question accurately.

CRITICAL SAFETY RULE — DEFER MECHANISM:
If ANY of the following are true, you MUST respond with DEFER instead of answering:
  • The image is too blurry, smoky, or dark to identify structures confidently
  • The retrieved evidence contradicts what you see or is insufficient
  • You are not confident enough to guide a surgical decision safely
  • The anatomy is ambiguous and a wrong answer could cause patient harm

RESPONSE FORMAT (pick exactly one):
  ANSWER: [your concise answer] | CONFIDENCE: [high/medium/low]
  DEFER: [brief reason why you cannot answer safely]

{context_block}

Be concise. Prioritize patient safety over completeness."""


def build_system_prompt(retrieved_chunks: list[tuple[dict, float]]) -> str:
    if not retrieved_chunks:
        context_block = "No retrieved evidence available."
    else:
        pieces = []
        for i, (chunk, score) in enumerate(retrieved_chunks):
            pieces.append(
                f"[Evidence {i+1} — {chunk['doc_title']} "
                f"(relevance {score:.2f})]:\n{chunk['text']}"
            )
        context_block = (
            "RETRIEVED SURGICAL KNOWLEDGE:\n"
            + "\n\n".join(pieces)
        )
    return SYSTEM_TEMPLATE.format(context_block=context_block)


# ─── VLM call ───────────────────────────────────────────────────────

def call_vlm(
    system_prompt: str,
    question: str,
    image_path: str | Path,
    mime: str = "image/jpeg",
    model: str = VLM_MODEL,
) -> str:
    """Call either OpenAI VLM or a local Hugging Face VLM."""
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
    """Call OpenAI-compatible VLM with image + text."""
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
    global _LOCAL_VLM, _LOCAL_PROCESSOR, _LOCAL_DEVICE
    if _LOCAL_VLM is not None and _LOCAL_PROCESSOR is not None:
        return _LOCAL_VLM, _LOCAL_PROCESSOR, _LOCAL_DEVICE

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(
        LOCAL_VLM_MODEL,
        cache_dir=HF_CACHE_DIR or None,
        local_files_only=HF_LOCAL_FILES_ONLY,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    if "florence-2" in LOCAL_VLM_MODEL.lower():
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_VLM_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=HF_CACHE_DIR or None,
            local_files_only=HF_LOCAL_FILES_ONLY,
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            LOCAL_VLM_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=HF_CACHE_DIR or None,
            local_files_only=HF_LOCAL_FILES_ONLY,
            token=HF_TOKEN or None,
        )
    model.to(device)
    model.eval()

    _LOCAL_VLM = model
    _LOCAL_PROCESSOR = processor
    _LOCAL_DEVICE = device
    return _LOCAL_VLM, _LOCAL_PROCESSOR, _LOCAL_DEVICE


def call_local_hf_vlm(
    system_prompt: str,
    question: str,
    image_path: str | Path,
) -> str:
    import torch
    from PIL import Image

    model, processor, device = _load_local_hf_vlm()
    image = Image.open(image_path).convert("RGB")

    if "florence-2" in LOCAL_VLM_MODEL.lower():
        task = "<VQA>"
        florence_prompt = f"{task}{question}"
        inputs = processor(
            text=florence_prompt,
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

        generated_text = processor.batch_decode(
            generated,
            skip_special_tokens=False,
        )[0]

        try:
            parsed = processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(image.width, image.height),
            )
            if isinstance(parsed, dict):
                answer = parsed.get(task)
                if isinstance(answer, list) and answer:
                    answer = answer[0]
                if answer is None and parsed:
                    answer = next(iter(parsed.values()))
            else:
                answer = parsed
        except Exception:
            answer = generated_text

        answer = str(answer).strip() if answer is not None else ""
        if not answer:
            return "DEFER: local Florence model could not produce a reliable answer."
        return f"ANSWER: {answer} | CONFIDENCE: low"

    prompt = f"{system_prompt}\n\nQUESTION:\n{question}"
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
    return text.strip()


# ─── Response parsing ───────────────────────────────────────────────

def parse_response(raw: str) -> dict:
    """Extract structured fields from the VLM's raw text response."""
    is_defer = raw.upper().startswith("DEFER")

    confidence = "unknown"
    for level in ("high", "medium", "low"):
        if f"confidence: {level}" in raw.lower():
            confidence = level
            break

    # Extract the answer or defer reason
    answer_text = raw
    if is_defer:
        # Try to extract reason after "DEFER:"
        if ":" in raw:
            answer_text = raw.split(":", 1)[1].strip()
    else:
        # Try to extract answer before "| CONFIDENCE"
        if "ANSWER:" in raw.upper():
            answer_text = raw.split("ANSWER:", 1)[1] if "ANSWER:" in raw else raw
            if "|" in answer_text:
                answer_text = answer_text.split("|")[0].strip()
            answer_text = answer_text.strip()

    return {
        "raw_response": raw,
        "is_defer": is_defer,
        "confidence": confidence,
        "parsed_answer": answer_text,
    }


# ─── Single question pipeline ───────────────────────────────────────

def run_single(
    frame_path: Path,
    question: str,
    retriever: SurgicalRetriever,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict:
    """Run retrieval → prompt → VLM → parse for one question."""

    # 1) Retrieve
    retrieved = retriever.retrieve_hybrid(question, top_k=top_k)

    # 2) Build prompt
    system_prompt = build_system_prompt(retrieved)

    # 3) Call VLM
    mime = detect_mime(frame_path)
    raw = call_vlm(system_prompt, question, frame_path, mime)

    # 5) Parse
    parsed = parse_response(raw)

    return {
        "frame": str(frame_path.name),
        "question": question,
        "retrieved_chunks": [c["chunk_id"] for c, _ in retrieved],
        "retrieved_scores": [round(s, 3) for _, s in retrieved],
        "retrieved_previews": [c["text"][:200] for c, _ in retrieved],
        **parsed,
    }


# ─── Batch run ───────────────────────────────────────────────────────

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
    total_cost_est = 0.0

    print(f"\n{'='*60}")
    print(f"Running RAG-VQA pipeline on {len(questions)} questions")
    print(f"VLM: {VLM_PROVIDER}:{VLM_MODEL}  |  Retrieval top-k: {RETRIEVAL_TOP_K}")
    print(f"{'='*60}\n")

    for q in questions:
        qid = q["qid"]
        frame_path = FRAMES_DIR / q["frame"]

        # Check frame exists
        if not frame_path.exists():
            print(f"  ⚠ {qid}: Frame not found → {frame_path}  (skipping)")
            results.append({
                "qid": qid,
                "question": q["question"],
                "error": f"Frame not found: {frame_path}",
                "gold_answer": q["gold_answer"],
                "should_defer": q["should_defer"],
                "question_type": q["question_type"],
            })
            continue

        print(f"  {qid}: {q['question'][:55]}...", end="", flush=True)
        t0 = time.time()

        try:
            result = run_single(frame_path, q["question"], retriever)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            result = {
                "raw_response": "",
                "is_defer": False,
                "confidence": "unknown",
                "parsed_answer": f"ERROR: {e}",
                "retrieved_chunks": [],
            }

        elapsed = time.time() - t0

        # Attach metadata
        result["qid"] = qid
        result["gold_answer"] = q["gold_answer"]
        result["should_defer"] = q["should_defer"]
        result["question_type"] = q["question_type"]
        result["difficulty"] = q["difficulty"]
        result["latency_s"] = round(elapsed, 2)
        results.append(result)

        status = "DEFER" if result.get("is_defer") else f"ANS ({result.get('confidence', '?')})"
        print(f"  → {status}  [{elapsed:.1f}s]")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(results)} results → {output_path}")
    answered = sum(1 for r in results if not r.get("is_defer") and "error" not in r)
    deferred = sum(1 for r in results if r.get("is_defer"))
    errors = sum(1 for r in results if "error" in r)
    print(f"  Answered: {answered}  |  Deferred: {deferred}  |  Errors: {errors}")
    print(f"{'='*60}")

    return results


# ─── Offline / mock mode (no API key) ───────────────────────────────

def run_mock(retriever: SurgicalRetriever):
    """
    Run retrieval only — useful when you don't have an OpenAI key yet.
    Shows what chunks would be retrieved for each question.
    """
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n{'='*60}")
    print("MOCK MODE — retrieval only, no VLM call")
    print(f"{'='*60}\n")

    mock_results = []
    for q in questions:
        retrieved = retriever.retrieve_hybrid(q["question"], top_k=RETRIEVAL_TOP_K)
        print(f"\n{q['qid']}: {q['question']}")
        for rank, (chunk, score) in enumerate(retrieved, 1):
            print(f"  #{rank} [{score:.3f}] {chunk['text'][:120]}...")

        mock_results.append({
            "qid": q["qid"],
            "question": q["question"],
            "retrieved_chunks": [c["chunk_id"] for c, _ in retrieved],
            "retrieved_scores": [round(s, 3) for _, s in retrieved],
            "gold_answer": q["gold_answer"],
            "should_defer": q["should_defer"],
            "question_type": q["question_type"],
            "mode": "mock_retrieval_only",
        })

    # Save mock results
    mock_path = RESULTS_FILE.parent / "mock_retrieval_results.json"
    with open(mock_path, "w", encoding="utf-8") as f:
        json.dump(mock_results, f, indent=2, ensure_ascii=False)
    print(f"\nMock results saved → {mock_path}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    retriever = SurgicalRetriever()

    if VLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        print("\n⚠  OPENAI_API_KEY not set — running in MOCK mode (retrieval only)")
        run_mock(retriever)
    else:
        if VLM_PROVIDER == "local_hf":
            print(f"[Preflight] Loading local VLM: {LOCAL_VLM_MODEL}")
            _load_local_hf_vlm()
        run_all(retriever)


if __name__ == "__main__":
    main()
