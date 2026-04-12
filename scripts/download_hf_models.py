"""
download_hf_models.py — Pre-download all Hugging Face models used by the project.

Downloads:
  1. Dense retrieval model   (sentence-transformers)
  2. Reranker model          (if USE_RERANKER=1)
  3. Local VLM               (pipeline answer model)
  4. Judge VLM               (evaluation judge model)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DENSE_MODEL_NAME,
    USE_RERANKER,
    RERANKER_MODEL_NAME,
    LOCAL_VLM_MODEL,
    JUDGE_VLM_MODEL,
    HF_CACHE_DIR,
    HF_TOKEN,
)


def download_dense(cache_dir: str | None) -> bool:
    print(f"\n[STEP 1/4] Downloading dense retrieval model: {DENSE_MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    try:
        SentenceTransformer(
            DENSE_MODEL_NAME,
            cache_folder=cache_dir,
            local_files_only=False,
            token=HF_TOKEN or None,
        )
        print("[DONE] Dense retrieval model cached.")
        return True
    except Exception as e:
        print(f"[WARN] Dense retrieval model failed: {e}")
        return False


def download_reranker(cache_dir: str | None) -> bool:
    if not USE_RERANKER:
        print("\n[STEP 2/4] Reranker: disabled, skipping.")
        return True

    print(f"\n[STEP 2/4] Downloading reranker model: {RERANKER_MODEL_NAME}")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    try:
        AutoTokenizer.from_pretrained(
            RERANKER_MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False,
            token=HF_TOKEN or None,
        )
        AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False,
            token=HF_TOKEN or None,
        )
        print("[DONE] Reranker model cached.")
        return True
    except Exception as e:
        print(f"[WARN] Reranker model failed: {e}")
        return False


def download_local_vlm(cache_dir: str | None) -> bool:
    print(f"\n[STEP 3/4] Downloading local VLM: {LOCAL_VLM_MODEL}")
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
    try:
        AutoProcessor.from_pretrained(
            LOCAL_VLM_MODEL,
            cache_dir=cache_dir,
            local_files_only=False,
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )
        if "florence-2" in LOCAL_VLM_MODEL.lower():
            AutoModelForCausalLM.from_pretrained(
                LOCAL_VLM_MODEL,
                cache_dir=cache_dir,
                local_files_only=False,
                low_cpu_mem_usage=True,
                token=HF_TOKEN or None,
                trust_remote_code=True,
            )
        else:
            AutoModelForImageTextToText.from_pretrained(
                LOCAL_VLM_MODEL,
                cache_dir=cache_dir,
                local_files_only=False,
                low_cpu_mem_usage=True,
                token=HF_TOKEN or None,
            )
        print("[DONE] Local VLM cached.")
        return True
    except Exception as e:
        print(f"[ERROR] Local VLM failed: {e}")
        return False


def download_judge_vlm(cache_dir: str | None) -> bool:
    print(f"\n[STEP 4/4] Downloading judge VLM: {JUDGE_VLM_MODEL}")

    is_qwen25_vl = "qwen2.5" in JUDGE_VLM_MODEL.lower() and "vl" in JUDGE_VLM_MODEL.lower()
    is_qwen_vl   = "qwen" in JUDGE_VLM_MODEL.lower() and "vl" in JUDGE_VLM_MODEL.lower()

    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(
            JUDGE_VLM_MODEL,
            cache_dir=cache_dir,
            local_files_only=False,
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )

        if is_qwen25_vl:
            # ✅ Dùng class đúng cho Qwen2.5-VL
            from transformers import Qwen2_5_VLForConditionalGeneration
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                JUDGE_VLM_MODEL,
                cache_dir=cache_dir,
                local_files_only=False,
                low_cpu_mem_usage=True,
                token=HF_TOKEN or None,
                trust_remote_code=True,
            )
        elif is_qwen_vl:
            from transformers import Qwen2VLForConditionalGeneration
            Qwen2VLForConditionalGeneration.from_pretrained(
                JUDGE_VLM_MODEL,
                cache_dir=cache_dir,
                local_files_only=False,
                low_cpu_mem_usage=True,
                token=HF_TOKEN or None,
                trust_remote_code=True,
            )
        else:
            from transformers import AutoModelForImageTextToText
            AutoModelForImageTextToText.from_pretrained(
                JUDGE_VLM_MODEL,
                cache_dir=cache_dir,
                local_files_only=False,
                low_cpu_mem_usage=True,
                token=HF_TOKEN or None,
                trust_remote_code=True,
            )

        print("[DONE] Judge VLM cached.")
        return True
    except Exception as e:
        print(f"[ERROR] Judge VLM failed: {e}")
        return False


def main() -> None:
    cache_dir = HF_CACHE_DIR or None

    print("=" * 60)
    print("  Hugging Face Model Pre-Download")
    print("=" * 60)
    print(f"  cache_dir  = {cache_dir or '(default HF cache)'}")
    print(f"  dense      = {DENSE_MODEL_NAME}")
    print(f"  reranker   = {RERANKER_MODEL_NAME if USE_RERANKER else '(disabled)'}")
    print(f"  vlm        = {LOCAL_VLM_MODEL}")
    print(f"  judge      = {JUDGE_VLM_MODEL}")
    print("=" * 60)

    results = {
        "dense":    download_dense(cache_dir),
        "reranker": download_reranker(cache_dir),
        "vlm":      download_local_vlm(cache_dir),
        "judge":    download_judge_vlm(cache_dir),
    }

    print("\n" + "=" * 60)
    all_ok = all(results.values())
    if all_ok:
        print("[SUCCESS] All models cached locally.")
        print("[NEXT] You can run with HF_LOCAL_FILES_ONLY=1 for offline use.")
    else:
        print("[PARTIAL] Some models did not download:")
        for name, ok in results.items():
            status = "✓" if ok else "✗"
            print(f"  {status} {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()