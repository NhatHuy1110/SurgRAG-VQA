"""
run_all.py — One-click orchestrator: build corpus → retrieval test → VQA → evaluate.

Usage:
    python scripts/run_all.py              # full pipeline (needs OPENAI_API_KEY)
    python scripts/run_all.py --mock       # retrieval only, no API calls
    python scripts/run_all.py --eval-only  # just re-run evaluation on existing results
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_script(name: str, description: str):
    script = SCRIPTS_DIR / name
    print(f"\n{'━'*60}")
    print(f"  Step: {description}")
    print(f"  Running: {script}")
    print(f"{'━'*60}\n")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SCRIPTS_DIR.parent),  # project root
    )
    if result.returncode != 0:
        print(f"\n✗ {name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n✓ {description} — done\n")


def main():
    parser = argparse.ArgumentParser(description="Surgical RAG-VQA — run full pipeline")
    parser.add_argument("--mock", action="store_true",
                        help="Run retrieval only (no VLM API calls)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on existing results")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="Skip corpus building (reuse existing chunks)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Surgical RAG-VQA — Feasibility Spike Pipeline     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.eval_only:
        run_script("evaluate.py", "Evaluation")
        return

    if not args.skip_corpus:
        run_script("build_corpus.py", "Build RAG corpus (PDF → chunks)")

    if args.mock:
        # For mock mode, we import and run directly to use mock function
        print(f"\n{'━'*60}")
        print(f"  Step: Mock retrieval (no VLM)")
        print(f"{'━'*60}\n")

        sys.path.insert(0, str(SCRIPTS_DIR))
        from rag_vqa_pipeline import run_mock
        from retrieval import SurgicalRetriever
        retriever = SurgicalRetriever()
        run_mock(retriever)
        print("\n✓ Mock retrieval — done\n")
    else:
        run_script("rag_vqa_pipeline.py", "RAG-VQA pipeline (retrieval + VLM)")

    run_script("evaluate.py", "Evaluation & report generation")

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                    ✓ All steps complete                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\nCheck results/ folder for outputs:")
    print("  • spike_results_v1.json    — raw pipeline results")
    print("  • evaluation_report.md     — report for professor")
    print("  • metrics.json             — structured metrics")


if __name__ == "__main__":
    main()
