# Surgical RAG-VQA

Safety-aware retrieval-augmented visual question answering for laparoscopic cholecystectomy.

This repository implements a surgical RAG-VQA pipeline that combines:

- laparoscopic frame input
- retrieval from curated surgical guideline and anatomy documents
- a local Hugging Face vision-language model
- an explicit `DEFER` mechanism for unsafe or uncertain cases

The current codebase is aligned with the final experiment setup:

- frames: `data/frames_v3/`
- questions: `data/annotations/questions_v3.json`
- retrieval evaluation: `data/annotations/retrieval_eval_v3.json`
- retrieval corpus: `docs/chunks/chunks_v3.jsonl`
- main output: `results/spike_results_v3.json`

## Pipeline Overview

The active workflow in this repository has four stages:

1. Build a structured surgical retrieval corpus from guideline PDFs.
2. Retrieve relevant evidence using `retrieval.py` over `chunks_v3.jsonl`.
3. Run a local Hugging Face VLM on frame + retrieved evidence with a safety-aware prompt.
4. Evaluate answer/defer behavior and save structured outputs.

## Active Components

The main scripts used by the current experiment are:

- `scripts/build_corpus.py`
- `scripts/generate_annotations.py`
- `scripts/retrieval.py`
- `scripts/rag_vqa_pipeline.py`
- `scripts/evaluate.py`
- `scripts/download_hf_models.py`
- `scripts/config.py`

## Retrieval Design

The current retrieval stack is built around `retrieval.py` and includes:

- child-first retrieval over `chunks_v3.jsonl`
- parent-expanded evidence packaging for prompt construction
- field-aware BM25 over contextualized chunk fields
- dense retrieval with sentence-transformer embeddings
- optional neural reranking
- question-type-conditioned priors
- class-conditioned query expansion from detected visual classes
- adaptive evidence selection to reduce redundancy

## VLM Design

The current VLM path in `scripts/rag_vqa_pipeline.py` supports:

- `local_hf`
- `openai`
- `mock_vlm`

The recommended server configuration for the current project is `local_hf` with a GPU-backed Hugging Face model. The default local VLM in code is:

```env
LOCAL_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

The prompt is explicitly safety-oriented and instructs the model to return `DEFER` when:

- the frame is visually unreliable
- anatomy is ambiguous
- retrieved evidence is insufficient
- answering would be unsafe

## Server Setup

Use a fresh Python environment on the server. Do not copy a local virtual environment.

```bash
git clone <your-repo-url>
cd surg-rag-vqa
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

If the server uses CUDA, install the correct PyTorch build first, then install the remaining dependencies.

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Then create the environment file:

```bash
cp .env.example .env
```

## Recommended Server Configuration

Edit `.env` for the local Hugging Face pipeline.

```env
VLM_PROVIDER=local_hf
HF_TOKEN=hf_your_read_token_here
HF_CACHE_DIR=
HF_LOCAL_FILES_ONLY=0
LOCAL_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
LOCAL_VLM_MAX_NEW_TOKENS=256
RETRIEVAL_MODE=hybrid
DENSE_MODEL_NAME=BAAI/bge-large-en-v1.5
USE_RERANKER=1
RERANKER_MODEL_NAME=BAAI/bge-reranker-large
RERANK_TOP_N=20
```

Optional alternative VLM if your server stack supports it cleanly:

```env
LOCAL_VLM_MODEL=llava-hf/llava-1.5-7b-hf
```

## Prepare Models on Server

Pre-download the retrieval and VLM models before the first full run:

```bash
python scripts/download_hf_models.py
```

After the cache is fully prepared, you may switch to offline-only loading:

```bash
export HF_LOCAL_FILES_ONLY=1
```

## Run on Server

### 1. Build the retrieval corpus

Run this when `docs/raw/` changes or when you want to regenerate the `v3` chunk corpus.

```bash
python scripts/build_corpus.py
```

### 2. Run the full RAG-VQA pipeline

```bash
python scripts/rag_vqa_pipeline.py
```

This writes the main structured output to:

- `results/spike_results_v3.json`

### 3. Evaluate results

```bash
python scripts/evaluate.py
```

## Data and Artifact Notes

This repository ignores generated data and artifacts such as:

- `data/`
- `docs/chunks/`
- `docs/raw/`
- `results/`

That means a server pull of code alone may not be sufficient if the required data or chunk files are not already present. In practice, the server must have:

- the `frames` images
- the `questions_v3` and `retrieval_eval_v3` annotation files
- the source PDFs in `docs/raw/` or a prebuilt `chunks_v3.jsonl`

## Security and Operational Notes

- Do not commit a real `.env` file.
- Do not expose a real Hugging Face token in a public repository.
- Install a CUDA-compatible PyTorch build if you plan to run local HF models on GPU.
- If the first local HF run fails, check model caching, GPU visibility, and Transformers compatibility before rerunning the full pipeline.

## Summary

This repository is a research-oriented surgical RAG-VQA prototype with:

- a structured `v3` frame and annotation setup
- a hierarchical retrieval corpus
- retrieval optimized for surgical question types
- a local Hugging Face VLM pipeline with explicit defer behavior
- evaluation focused on safety-aware answer vs defer performance
