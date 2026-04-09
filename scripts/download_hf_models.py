import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DENSE_MODEL_NAME, LOCAL_VLM_MODEL, HF_CACHE_DIR, HF_TOKEN


def main() -> None:
    cache_dir = HF_CACHE_DIR or None

    print("[INFO] Hugging Face model pre-download")
    print(f"[INFO] cache_dir = {cache_dir or '(default Hugging Face cache)'}")
    print(f"[INFO] dense model = {DENSE_MODEL_NAME}")
    print(f"[INFO] vlm model   = {LOCAL_VLM_MODEL}")

    print("\n[STEP] Downloading dense retrieval model...")
    from sentence_transformers import SentenceTransformer
    dense_ok = False
    try:
        SentenceTransformer(
            DENSE_MODEL_NAME,
            cache_folder=cache_dir,
            local_files_only=False,
            token=HF_TOKEN or None,
        )
        dense_ok = True
        print("[DONE] Dense retrieval model is cached.")
    except Exception as e:
        print(f"[WARN] Dense retrieval model download failed: {e}")
        print("[WARN] Continuing to the VLM download step.")

    print("\n[STEP] Downloading local VLM...")
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
    vlm_ok = False
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
        vlm_ok = True
        print("[DONE] Local VLM is cached.")
    except Exception as e:
        print(f"[ERROR] Local VLM download failed: {e}")

    if dense_ok and vlm_ok:
        print("\n[SUCCESS] All required Hugging Face models are cached locally.")
        print("[NEXT] You can run with HF_LOCAL_FILES_ONLY=1 for offline use.")
        return

    print("\n[PARTIAL] Model download did not fully complete.")
    if not dense_ok:
        print("[PARTIAL] Dense retrieval model is not confirmed by this run.")
    if not vlm_ok:
        print("[PARTIAL] Local VLM is not confirmed by this run.")


if __name__ == "__main__":
    main()
