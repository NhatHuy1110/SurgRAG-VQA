"""
run_all.py - Build the corpus, run the pipeline, and evaluate results.
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent


def run_script(name: str, description: str) -> None:
    script = SCRIPTS_DIR / name
    print(f"\n{'=' * 60}")
    print(f"Step: {description}")
    print(f"Running: {script.name}")
    print(f"{'=' * 60}\n")

    result = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] {name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"[DONE] {description}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Surgical RAG-VQA pipeline.")
    parser.add_argument("--mock", action="store_true", help="Run retrieval only without calling a VLM")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing results")
    parser.add_argument("--skip-corpus", action="store_true", help="Reuse the existing chunk file")
    args = parser.parse_args()

    print("Surgical RAG-VQA Pipeline")

    if args.eval_only:
        run_script("evaluate.py", "Evaluate results")
        return

    if not args.skip_corpus:
        run_script("build_corpus.py", "Build retrieval corpus")

    if args.mock:
        print(f"\n{'=' * 60}")
        print("Step: Mock retrieval")
        print(f"{'=' * 60}\n")
        sys.path.insert(0, str(SCRIPTS_DIR))
        from rag_vqa_pipeline import run_mock
        from retrieval import SurgicalRetriever

        retriever = SurgicalRetriever()
        run_mock(retriever)
        print("[DONE] Mock retrieval\n")
    else:
        run_script("rag_vqa_pipeline.py", "Run RAG-VQA pipeline")

    run_script("evaluate.py", "Evaluate results")

    print("Outputs:")
    print("- results/spike_results.json")
    print("- results/evaluation_report.md")
    print("- results/metrics.json")


if __name__ == "__main__":
    main()
