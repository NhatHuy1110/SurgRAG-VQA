# Pilot 1: Observability Predictive Validity

This pilot tests whether pre-generation observability predicts CVS-VQA accuracy on Endoscapes-CVS201 frames.

It is intentionally isolated from the existing RAG-VQA code. It only reads:

- `data/endoscapes_work/registry_v0.jsonl`
- `data/endoscapes/`

and writes:

- `pilots/pilot1_observability/outputs/pilot1/`

## Core Design

1. Compute visual observability for all 11,090 Endoscapes-CVS201 frames.
2. Split the full registry into global tertiles: `low`, `mid`, `high`.
3. Sample 100 frames from each visual stratum, with best-effort CVS state balance.
4. Ask each VLM three official CVS criterion questions.
5. Score strict accuracy and coverage-aware answered accuracy by observability stratum.

Official Endoscapes mapping:

- `c1`: two and only two structures entering the gallbladder
- `c2`: hepatocystic triangle clearance
- `c3`: lower gallbladder / cystic plate exposure

## Recommended Models

Primary:

- `Qwen/Qwen2.5-VL-7B-Instruct`

Optional robustness checks:

- `Qwen/Qwen2-VL-7B-Instruct`
- `llava-hf/llava-1.5-7b-hf`

Run one model first. Enable extra models in `config.yaml` after the first run works.

## Server Setup

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# Pick the CUDA wheel matching the server.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install transformers accelerate qwen-vl-utils pillow opencv-python pandas numpy scipy scikit-learn matplotlib tqdm pyyaml
```

## Run

Check GPU:

```bash
python pilots/pilot1_observability/scripts/00_check_gpu.py
```

Build global observability features and the 300-frame pilot manifest:

```bash
python pilots/pilot1_observability/scripts/01_build_manifest.py \
  --config pilots/pilot1_observability/config.yaml
```

Smoke test one model on 5 frames:

```bash
python pilots/pilot1_observability/scripts/02_run_vlm_cvs.py \
  --config pilots/pilot1_observability/config.yaml \
  --model-id qwen25_vl_7b \
  --limit 5
```

Run full 300-frame pilot for one model:

```bash
python pilots/pilot1_observability/scripts/02_run_vlm_cvs.py \
  --config pilots/pilot1_observability/config.yaml \
  --model-id qwen25_vl_7b
```

Score and plot:

```bash
python pilots/pilot1_observability/scripts/03_score_and_plot.py \
  --config pilots/pilot1_observability/config.yaml
```

Main outputs:

- `pilot_manifest.csv`
- `obs_features_all_registry.csv`
- `predictions/<model_id>.jsonl`
- `scored_predictions.csv`
- `summary_by_stratum.csv`
- `summary_by_criterion.csv`
- `figures/fig_accuracy_by_visual_observability_<model_id>.png`
- `pilot1_memo.md`

