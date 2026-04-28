from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


CVS_STATE_ORDER = ["no", "partial", "achieved"]
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


def resolve_frame_path(row: dict[str, Any], endoscapes_root: Path) -> str:
    raw = Path(str(row.get("frame_path") or ""))
    if raw.exists():
        return str(raw)

    split = str(row.get("split") or "")
    image_filename = str(row.get("image_filename") or "")
    if split and image_filename:
        candidate = endoscapes_root / split / image_filename
        if candidate.exists():
            return str(candidate)

    video_id = str(row.get("video_id") or "")
    frame_id = row.get("frame_id")
    if split and video_id and frame_id is not None:
        candidate = endoscapes_root / split / f"{video_id}_{int(frame_id)}.jpg"
        if candidate.exists():
            return str(candidate)

    return str(raw)


def robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    med = np.nanmedian(values)
    iqr = np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
    if not np.isfinite(iqr) or iqr < 1e-8:
        iqr = np.nanstd(values)
    if not np.isfinite(iqr) or iqr < 1e-8:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - med) / iqr


def image_features(frame_path: str) -> dict[str, float]:
    img = cv2.imread(frame_path)
    if img is None:
        return {
            "sharpness_lap_var": math.nan,
            "brightness_mean": math.nan,
            "contrast_std": math.nan,
            "underexposed_frac": math.nan,
            "overexposed_frac": math.nan,
            "specular_frac": math.nan,
            "red_dominance_frac": math.nan,
            "height": math.nan,
            "width": math.nan,
        }

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img)
    _, s, v = cv2.split(hsv)

    return {
        "sharpness_lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "underexposed_frac": float(np.mean(gray < 15)),
        "overexposed_frac": float(np.mean(gray > 245)),
        "specular_frac": float(np.mean((v > 245) & (s < 60))),
        "red_dominance_frac": float(np.mean((r > 80) & (r > 1.25 * g) & (r > 1.25 * b))),
        "height": float(h),
        "width": float(w),
    }


def cvs_state(majority_positive_count: int) -> str:
    if majority_positive_count <= 0:
        return "no"
    if majority_positive_count >= 3:
        return "achieved"
    return "partial"


def build_observability(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["sharpness_log"] = np.log1p(out["sharpness_lap_var"])
    out["brightness_good"] = 1.0 - np.minimum((out["brightness_mean"] - 128.0).abs() / 128.0, 1.0)
    out["contrast_good"] = np.minimum(out["contrast_std"] / 64.0, 1.0)
    out["exposure_bad"] = out["underexposed_frac"].fillna(0) + out["overexposed_frac"].fillna(0)

    labels_count = out["spatial_labels"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    out["spatial_signal"] = (
        0.50 * out["insseg_available"].astype(float)
        + 0.35 * out["semseg_available"].astype(float)
        + 0.15 * np.minimum(labels_count / 6.0, 1.0)
    )

    feature_cols = [
        "sharpness_log",
        "brightness_good",
        "contrast_good",
        "exposure_bad",
        "specular_frac",
        "red_dominance_frac",
        "spatial_signal",
        "disagreement_count",
        "disagreement_score",
    ]
    for col in feature_cols:
        out[f"z_{col}"] = robust_z(out[col])

    weights = cfg["observability"]["weights"]
    out["obs_visual_score"] = 0.0
    for col, weight in weights.items():
        out["obs_visual_score"] += float(weight) * out[f"z_{col}"].fillna(0.0)

    alpha = float(cfg["observability"].get("construct_disagreement_alpha", 0.30))
    out["obs_construct_score"] = out["obs_visual_score"] - alpha * out["z_disagreement_count"].fillna(0.0)

    for score_col, stratum_col in [
        ("obs_visual_score", "obs_visual_stratum"),
        ("obs_construct_score", "obs_construct_stratum"),
    ]:
        rank = out[score_col].rank(method="first")
        out[stratum_col] = pd.qcut(rank, q=3, labels=OBS_ORDER)

    return out


def sample_pilot(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    seed = int(cfg["sampling"]["random_seed"])
    n_per_obs = int(cfg["sampling"]["n_per_observability_stratum"])
    balance = bool(cfg["sampling"].get("balance_cvs_state", True))
    targets = cfg["sampling"].get("cvs_state_targets", {"no": 33, "partial": 34, "achieved": 33})

    rng = np.random.default_rng(seed)
    samples = []

    for obs in OBS_ORDER:
        obs_df = df[df["obs_visual_stratum"].astype(str) == obs].copy()
        taken_ids: set[str] = set()
        obs_parts = []

        if balance:
            for state in CVS_STATE_ORDER:
                target = int(targets.get(state, 0))
                cell = obs_df[obs_df["cvs_state"] == state]
                n = min(target, len(cell))
                if n > 0:
                    part = cell.sample(n=n, random_state=seed + len(samples) + len(obs_parts))
                    obs_parts.append(part)
                    taken_ids.update(part["record_id"].astype(str))

        obs_sample = pd.concat(obs_parts, ignore_index=False) if obs_parts else pd.DataFrame(columns=obs_df.columns)
        missing = n_per_obs - len(obs_sample)
        if missing > 0:
            remaining = obs_df[~obs_df["record_id"].astype(str).isin(taken_ids)]
            fill_n = min(missing, len(remaining))
            if fill_n > 0:
                obs_sample = pd.concat(
                    [obs_sample, remaining.sample(n=fill_n, random_state=int(rng.integers(1, 1_000_000)))],
                    ignore_index=False,
                )

        if len(obs_sample) > n_per_obs:
            obs_sample = obs_sample.sample(n=n_per_obs, random_state=seed)
        samples.append(obs_sample)

    pilot = pd.concat(samples, ignore_index=True)
    pilot = pilot.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pilot["pilot_row_index"] = np.arange(len(pilot))
    return pilot


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    root = Path(config_path).resolve().parents[2]
    registry_path = root / cfg["paths"]["registry_jsonl"]
    endoscapes_root = root / cfg["paths"]["endoscapes_root"]
    output_dir = root / cfg["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_jsonl(registry_path)
    rows = []
    for row in registry:
        frame_path = resolve_frame_path(row, endoscapes_root)
        rows.append(
            {
                "record_id": row["record_id"],
                "split": row.get("split"),
                "video_id": row.get("video_id"),
                "frame_id": row.get("frame_id"),
                "image_filename": row.get("image_filename"),
                "frame_path": frame_path,
                "frame_exists": Path(frame_path).exists(),
                "c1": int(float(row.get("c1", 0))),
                "c2": int(float(row.get("c2", 0))),
                "c3": int(float(row.get("c3", 0))),
                "majority_positive_count": int(row.get("majority_positive_count", 0)),
                "cvs_achieved_majority": bool(row.get("cvs_achieved_majority")),
                "agreement_score": float(row.get("agreement_score", 0.0)),
                "disagreement_score": float(row.get("disagreement_score", 0.0)),
                "disagreement_count": int(row.get("disagreement_count", 0)),
                "insseg_available": bool(row.get("insseg_available")),
                "semseg_available": bool(row.get("semseg_available")),
                "spatial_labels": row.get("spatial_labels") or [],
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["frame_exists"]].copy().reset_index(drop=True)
    df["cvs_state"] = df["majority_positive_count"].apply(cvs_state)

    features = []
    for path in tqdm(df["frame_path"], desc="Computing image features"):
        features.append(image_features(path))
    feat_df = pd.DataFrame(features)
    df = pd.concat([df, feat_df], axis=1)
    df = build_observability(df, cfg)

    all_path = output_dir / "obs_features_all_registry.csv"
    df.to_csv(all_path, index=False)

    pilot = sample_pilot(df, cfg)
    manifest_path = output_dir / "pilot_manifest.csv"
    pilot.to_csv(manifest_path, index=False)

    crosstab = pd.crosstab(pilot["obs_visual_stratum"], pilot["cvs_state"])
    crosstab_path = output_dir / "pilot_sampling_crosstab.csv"
    crosstab.to_csv(crosstab_path)

    print(f"Saved all registry features: {all_path}")
    print(f"Saved pilot manifest: {manifest_path}")
    print(f"Saved sampling crosstab: {crosstab_path}")
    print("\nObservability strata in pilot:")
    print(pilot["obs_visual_stratum"].value_counts())
    print("\nCVS state by visual stratum:")
    print(crosstab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("pilots/pilot1_observability/config.yaml"))
    args = parser.parse_args()
    main(args.config)

