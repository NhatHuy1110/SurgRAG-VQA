from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


OBS_ORDER = ["low", "mid", "high"]


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def answer_to_binary(answer: str) -> float:
    answer = str(answer).lower().strip()
    if answer == "yes":
        return 1.0
    if answer == "no":
        return 0.0
    return np.nan


def bootstrap_ci(values: pd.Series, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_predictions(manifest: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(
        manifest,
        left_on="pilot_row_index",
        right_on="pilot_row_index",
        suffixes=("", "_manifest"),
        how="left",
    )
    gt_values = []
    pred_values = []
    strict_correct = []
    answered_correct = []
    answered_mask = []

    for _, row in merged.iterrows():
        criterion = str(row["criterion"])
        gt = int(row[criterion])
        pred = answer_to_binary(row["parsed_answer"])
        gt_values.append(gt)
        pred_values.append(pred)
        if np.isnan(pred):
            strict_correct.append(0)
            answered_correct.append(np.nan)
            answered_mask.append(0)
        else:
            corr = int(int(pred) == gt)
            strict_correct.append(corr)
            answered_correct.append(corr)
            answered_mask.append(1)

    merged["gt"] = gt_values
    merged["pred_bin"] = pred_values
    merged["strict_correct"] = strict_correct
    merged["answered_correct"] = answered_correct
    merged["answered"] = answered_mask
    merged["uncertain_or_error"] = 1 - merged["answered"]
    return merged


def summarize(scored: pd.DataFrame, stratum_col: str, score_type: str) -> pd.DataFrame:
    rows = []
    for model_id in sorted(scored["model_id"].unique()):
        model_rows = scored[scored["model_id"] == model_id]
        for stratum in OBS_ORDER:
            sub = model_rows[model_rows[stratum_col].astype(str) == stratum]
            if sub.empty:
                continue
            strict_mean, strict_lo, strict_hi = bootstrap_ci(sub["strict_correct"])
            answered = sub[sub["answered"] == 1]
            ans_mean, ans_lo, ans_hi = bootstrap_ci(answered["answered_correct"])
            rows.append(
                {
                    "model_id": model_id,
                    "score_type": score_type,
                    "stratum": stratum,
                    "n_predictions": len(sub),
                    "n_frames": sub["record_id"].nunique(),
                    "strict_accuracy": strict_mean,
                    "strict_ci_low": strict_lo,
                    "strict_ci_high": strict_hi,
                    "answered_accuracy": ans_mean,
                    "answered_ci_low": ans_lo,
                    "answered_ci_high": ans_hi,
                    "uncertain_rate": float(sub["uncertain_or_error"].mean()),
                    "yes_rate": float((sub["parsed_answer"] == "yes").mean()),
                    "no_rate": float((sub["parsed_answer"] == "no").mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_by_criterion(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, criterion, stratum), sub in scored.groupby(["model_id", "criterion", "obs_visual_stratum"]):
        strict_mean, strict_lo, strict_hi = bootstrap_ci(sub["strict_correct"])
        rows.append(
            {
                "model_id": model_id,
                "criterion": criterion,
                "obs_visual_stratum": str(stratum),
                "n": len(sub),
                "strict_accuracy": strict_mean,
                "strict_ci_low": strict_lo,
                "strict_ci_high": strict_hi,
                "uncertain_rate": float(sub["uncertain_or_error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for score_type in sorted(summary["score_type"].unique()):
        for model_id in sorted(summary["model_id"].unique()):
            sub = summary[(summary["score_type"] == score_type) & (summary["model_id"] == model_id)].copy()
            sub["stratum"] = pd.Categorical(sub["stratum"], categories=OBS_ORDER, ordered=True)
            sub = sub.sort_values("stratum")
            if sub.empty:
                continue

            x = np.arange(len(sub))
            y = sub["strict_accuracy"].to_numpy()
            yerr = np.vstack([
                y - sub["strict_ci_low"].to_numpy(),
                sub["strict_ci_high"].to_numpy() - y,
            ])

            plt.figure(figsize=(7, 5))
            plt.bar(x, y)
            plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5, color="black")
            plt.xticks(x, sub["stratum"].astype(str).to_numpy())
            plt.ylim(0, 1)
            plt.ylabel("Strict CVS-VQA accuracy")
            plt.xlabel(f"{score_type} observability stratum")
            plt.title(f"Pilot 1: {model_id}")
            plt.tight_layout()
            plt.savefig(plot_dir / f"fig_accuracy_by_{score_type}_observability_{model_id}.png", dpi=200)
            plt.close()

    visual = summary[summary["score_type"] == "visual"].copy()
    if visual["model_id"].nunique() > 1:
        plt.figure(figsize=(8, 5))
        for model_id in sorted(visual["model_id"].unique()):
            sub = visual[visual["model_id"] == model_id].copy()
            sub["stratum"] = pd.Categorical(sub["stratum"], categories=OBS_ORDER, ordered=True)
            sub = sub.sort_values("stratum")
            plt.plot(sub["stratum"].astype(str), sub["strict_accuracy"], marker="o", label=model_id)
        plt.ylim(0, 1)
        plt.ylabel("Strict CVS-VQA accuracy")
        plt.xlabel("Visual observability stratum")
        plt.title("Pilot 1: model comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "fig_accuracy_by_visual_observability_model_comparison.png", dpi=200)
        plt.close()


def plot_score_distribution(manifest: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.hist(manifest["obs_visual_score"].dropna(), bins=30)
    plt.xlabel("obs_visual_score")
    plt.ylabel("Frames")
    plt.title("Pilot 1 manifest visual observability distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "fig_obs_score_distribution.png", dpi=200)
    plt.close()


def copy_examples(scored: pd.DataFrame, output_dir: Path) -> None:
    base = output_dir / "error_examples"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    for model_id in sorted(scored["model_id"].unique()):
        model_rows = scored[scored["model_id"] == model_id]
        groups = {
            "low_obs_wrong": model_rows[(model_rows["obs_visual_stratum"].astype(str) == "low") & (model_rows["strict_correct"] == 0)],
            "high_obs_wrong": model_rows[(model_rows["obs_visual_stratum"].astype(str) == "high") & (model_rows["strict_correct"] == 0)],
            "low_obs_correct": model_rows[(model_rows["obs_visual_stratum"].astype(str) == "low") & (model_rows["strict_correct"] == 1)],
            "high_obs_correct": model_rows[(model_rows["obs_visual_stratum"].astype(str) == "high") & (model_rows["strict_correct"] == 1)],
        }
        for group_name, rows in groups.items():
            folder = base / model_id / group_name
            folder.mkdir(parents=True, exist_ok=True)
            seen = set()
            count = 0
            for _, row in rows.iterrows():
                path = Path(str(row["frame_path_manifest"]))
                if path in seen or not path.exists():
                    continue
                seen.add(path)
                dst = folder / f"{count:02d}_{row['criterion']}_{path.name}"
                shutil.copy(path, dst)
                count += 1
                if count >= 10:
                    break


def verdict_for_model(summary: pd.DataFrame, model_id: str, score_type: str) -> tuple[str, float, bool]:
    sub = summary[(summary["model_id"] == model_id) & (summary["score_type"] == score_type)].set_index("stratum")
    if not all(s in sub.index for s in OBS_ORDER):
        return "INCOMPLETE", np.nan, False
    low = float(sub.loc["low", "strict_accuracy"])
    mid = float(sub.loc["mid", "strict_accuracy"])
    high = float(sub.loc["high", "strict_accuracy"])
    gap = high - low
    monotonic = high >= mid >= low
    if monotonic and gap >= 0.15:
        return "PASS", gap, monotonic
    if gap >= 0.10:
        return "MARGINAL", gap, monotonic
    return "FAIL", gap, monotonic


def write_memo(summary: pd.DataFrame, criterion_summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Pilot 1 Observability Memo",
        "",
        "Goal: test whether pre-generation observability predicts CVS-VQA accuracy.",
        "",
        "Official Endoscapes mapping used:",
        "- C1: two and only two structures entering the gallbladder",
        "- C2: hepatocystic triangle clearance",
        "- C3: lower gallbladder / cystic plate exposure",
        "",
        "## Visual Observability Verdicts",
        "",
        "| Model | Low | Mid | High | Gap High-Low | Monotonic | Verdict |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for model_id in sorted(summary["model_id"].unique()):
        sub = summary[(summary["model_id"] == model_id) & (summary["score_type"] == "visual")].set_index("stratum")
        verdict, gap, monotonic = verdict_for_model(summary, model_id, "visual")
        low = sub.loc["low", "strict_accuracy"] if "low" in sub.index else np.nan
        mid = sub.loc["mid", "strict_accuracy"] if "mid" in sub.index else np.nan
        high = sub.loc["high", "strict_accuracy"] if "high" in sub.index else np.nan
        lines.append(f"| {model_id} | {low:.3f} | {mid:.3f} | {high:.3f} | {gap:.3f} | {monotonic} | {verdict} |")

    lines.extend(
        [
            "",
            "## Construct Observability Check",
            "",
            "| Model | Low | Mid | High | Gap High-Low | Monotonic | Verdict |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for model_id in sorted(summary["model_id"].unique()):
        sub = summary[(summary["model_id"] == model_id) & (summary["score_type"] == "construct")].set_index("stratum")
        verdict, gap, monotonic = verdict_for_model(summary, model_id, "construct")
        low = sub.loc["low", "strict_accuracy"] if "low" in sub.index else np.nan
        mid = sub.loc["mid", "strict_accuracy"] if "mid" in sub.index else np.nan
        high = sub.loc["high", "strict_accuracy"] if "high" in sub.index else np.nan
        lines.append(f"| {model_id} | {low:.3f} | {mid:.3f} | {high:.3f} | {gap:.3f} | {monotonic} | {verdict} |")

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- PASS: high >= mid >= low and high-low >= 0.15.",
            "- MARGINAL: high-low >= 0.10 but trend is imperfect.",
            "- FAIL: high-low < 0.10 or clearly non-monotonic.",
            "- If construct passes but visual fails, the hypothesis may still be viable but visual features are weak.",
            "",
            "## Files",
            "",
            "- `scored_predictions.csv`",
            "- `summary_by_stratum.csv`",
            "- `summary_by_criterion.csv`",
            "- `figures/`",
            "- `error_examples/`",
        ]
    )
    (output_dir / "pilot1_memo.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("pilots/pilot1_observability/config.yaml"))
    parser.add_argument("--model-id", default="", help="Score one model id. If omitted, scores all prediction files.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(args.config).resolve().parents[2]
    output_dir = root / cfg["paths"]["output_dir"]
    manifest = pd.read_csv(output_dir / "pilot_manifest.csv")

    pred_dir = output_dir / "predictions"
    prediction_files = [pred_dir / f"{args.model_id}.jsonl"] if args.model_id else sorted(pred_dir.glob("*.jsonl"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")

    scored_parts = []
    for pred_file in prediction_files:
        preds = pd.DataFrame(load_jsonl(pred_file))
        if preds.empty:
            continue
        scored_parts.append(score_predictions(manifest, preds))
    scored = pd.concat(scored_parts, ignore_index=True)

    scored_path = output_dir / "scored_predictions.csv"
    scored.to_csv(scored_path, index=False)

    visual_summary = summarize(scored, "obs_visual_stratum", "visual")
    construct_summary = summarize(scored, "obs_construct_stratum", "construct")
    summary = pd.concat([visual_summary, construct_summary], ignore_index=True)
    summary_path = output_dir / "summary_by_stratum.csv"
    summary.to_csv(summary_path, index=False)

    criterion_summary = summarize_by_criterion(scored)
    criterion_summary_path = output_dir / "summary_by_criterion.csv"
    criterion_summary.to_csv(criterion_summary_path, index=False)

    plot_accuracy(summary, output_dir)
    plot_score_distribution(manifest, output_dir)
    copy_examples(scored, output_dir)
    write_memo(summary, criterion_summary, output_dir)

    print(f"Saved scored predictions: {scored_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved criterion summary: {criterion_summary_path}")
    print(f"Saved memo: {output_dir / 'pilot1_memo.md'}")


if __name__ == "__main__":
    main()

