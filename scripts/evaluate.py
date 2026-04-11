"""
evaluate.py - Evaluate Surgical RAG-VQA results.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIDENCE_LEVELS, RESULTS_DIR, RESULTS_FILE


def load_results(path: Path = RESULTS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(results: list[dict]) -> dict:
    valid = [row for row in results if "error" not in row]
    total = len(valid)
    if total == 0:
        return {"error": "No valid results to evaluate"}

    deferred = sum(1 for row in valid if row.get("is_defer"))
    answered = total - deferred

    defer_tp = sum(1 for row in valid if row["should_defer"] and row.get("is_defer"))
    defer_fp = sum(1 for row in valid if not row["should_defer"] and row.get("is_defer"))
    defer_fn = sum(1 for row in valid if row["should_defer"] and not row.get("is_defer"))
    defer_tn = sum(1 for row in valid if not row["should_defer"] and not row.get("is_defer"))

    defer_precision = defer_tp / (defer_tp + defer_fp) if (defer_tp + defer_fp) else 0.0
    defer_recall = defer_tp / (defer_tp + defer_fn) if (defer_tp + defer_fn) else 0.0
    defer_f1 = (
        2 * defer_precision * defer_recall / (defer_precision + defer_recall)
        if (defer_precision + defer_recall)
        else 0.0
    )

    confidence_dist = {level: 0 for level in CONFIDENCE_LEVELS}
    for row in valid:
        confidence = row.get("confidence", "unknown")
        confidence_dist[confidence] = confidence_dist.get(confidence, 0) + 1

    by_type = defaultdict(list)
    for row in valid:
        by_type[row["question_type"]].append(row)

    type_stats = {}
    for question_type, items in by_type.items():
        deferred_count = sum(1 for row in items if row.get("is_defer"))
        type_stats[question_type] = {
            "count": len(items),
            "deferred": deferred_count,
            "answered": len(items) - deferred_count,
        }

    by_difficulty = defaultdict(list)
    for row in valid:
        by_difficulty[row.get("difficulty", "unknown")].append(row)

    difficulty_stats = {}
    for difficulty, items in by_difficulty.items():
        difficulty_stats[difficulty] = {
            "count": len(items),
            "deferred": sum(1 for row in items if row.get("is_defer")),
        }

    latencies = [row.get("latency_s", 0) for row in valid if row.get("latency_s")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    per_question = []
    for row in valid:
        if row.get("is_defer"):
            status = "DEFER"
        else:
            status = f"ANS ({row.get('confidence', '?')})"

        if row["should_defer"] and row.get("is_defer"):
            defer_note = "correct defer"
        elif row["should_defer"] and not row.get("is_defer"):
            defer_note = "missed defer"
        elif not row["should_defer"] and row.get("is_defer"):
            defer_note = "unnecessary defer"
        else:
            defer_note = "correct answer"

        per_question.append(
            {
                "qid": row["qid"],
                "question_type": row["question_type"],
                "difficulty": row.get("difficulty", "?"),
                "status": status,
                "defer_note": defer_note,
                "parsed_answer": row.get("parsed_answer", "")[:120],
                "gold_answer": row.get("gold_answer", "")[:120],
            }
        )

    return {
        "total": total,
        "answered": answered,
        "deferred": deferred,
        "defer_tp": defer_tp,
        "defer_fp": defer_fp,
        "defer_fn": defer_fn,
        "defer_tn": defer_tn,
        "defer_precision": defer_precision,
        "defer_recall": defer_recall,
        "defer_f1": defer_f1,
        "confidence_dist": confidence_dist,
        "by_type": type_stats,
        "by_difficulty": difficulty_stats,
        "avg_latency_s": avg_latency,
        "per_question": per_question,
    }


def print_report(metrics: dict) -> None:
    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        return

    print()
    print("=" * 60)
    print("Surgical RAG-VQA Evaluation Report")
    print("=" * 60)
    print(f"Total questions: {metrics['total']}")
    print(f"Answered: {metrics['answered']}")
    print(f"Deferred: {metrics['deferred']}")
    print()
    print("Defer Metrics")
    print(f"- TP: {metrics['defer_tp']}")
    print(f"- FP: {metrics['defer_fp']}")
    print(f"- FN: {metrics['defer_fn']}")
    print(f"- TN: {metrics['defer_tn']}")
    print(f"- Precision: {metrics['defer_precision']:.2f}")
    print(f"- Recall: {metrics['defer_recall']:.2f}")
    print(f"- F1: {metrics['defer_f1']:.2f}")
    print()
    print("Confidence Distribution")
    for level, count in metrics["confidence_dist"].items():
        print(f"- {level}: {count}")
    print()
    print("By Question Type")
    for question_type, stats in metrics["by_type"].items():
        print(f"- {question_type}: {stats['count']} total, {stats['deferred']} deferred, {stats['answered']} answered")
    if metrics["avg_latency_s"]:
        print()
        print(f"Average latency: {metrics['avg_latency_s']:.1f}s")


def generate_markdown_report(metrics: dict, output_path: Path) -> None:
    if "error" in metrics:
        output_path.write_text(f"# Error\n\n{metrics['error']}\n", encoding="utf-8")
        return

    lines = [
        "# Surgical RAG-VQA Evaluation Report",
        "",
        "## Summary",
        f"- Total questions: {metrics['total']}",
        f"- Answered: {metrics['answered']}",
        f"- Deferred: {metrics['deferred']}",
        f"- Average latency: {metrics['avg_latency_s']:.1f}s" if metrics["avg_latency_s"] else "- Average latency: n/a",
        "",
        "## Defer Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| TP | {metrics['defer_tp']} |",
        f"| FP | {metrics['defer_fp']} |",
        f"| FN | {metrics['defer_fn']} |",
        f"| TN | {metrics['defer_tn']} |",
        f"| Precision | {metrics['defer_precision']:.2f} |",
        f"| Recall | {metrics['defer_recall']:.2f} |",
        f"| F1 | {metrics['defer_f1']:.2f} |",
        "",
        "## Confidence Distribution",
        "",
        "| Confidence | Count |",
        "|------------|-------|",
    ]

    for level, count in metrics["confidence_dist"].items():
        lines.append(f"| {level} | {count} |")

    lines += ["", "## By Question Type", "", "| Type | Count | Deferred | Answered |", "|------|-------|----------|----------|"]
    for question_type, stats in metrics["by_type"].items():
        lines.append(f"| {question_type} | {stats['count']} | {stats['deferred']} | {stats['answered']} |")

    lines += ["", "## Per Question", "", "| QID | Type | Difficulty | Status | Defer Check |", "|-----|------|------------|--------|-------------|"]
    for row in metrics["per_question"]:
        lines.append(f"| {row['qid']} | {row['question_type']} | {row['difficulty']} | {row['status']} | {row['defer_note']} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Markdown report saved -> {output_path}")


def main() -> None:
    results_path = RESULTS_FILE
    mock_path = RESULTS_DIR / "mock_retrieval_results.json"
    if not results_path.exists() and mock_path.exists():
        print("[WARN] No full results found, evaluating mock retrieval results instead.")
        results_path = mock_path

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("Run rag_vqa_pipeline.py first.")
        sys.exit(1)

    results = load_results(results_path)
    metrics = compute_metrics(results)
    print_report(metrics)

    markdown_path = RESULTS_DIR / "evaluation_report.md"
    generate_markdown_report(metrics, markdown_path)

    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_json = {key: value for key, value in metrics.items() if key != "per_question"}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"[DONE] Metrics JSON saved -> {metrics_path}")


if __name__ == "__main__":
    main()
