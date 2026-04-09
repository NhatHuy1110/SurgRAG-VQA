# Surgical RAG-VQA

Retrieval-augmented visual question answering for laparoscopic cholecystectomy.

This repo is organized around three stages:

1. Build a surgical text corpus from guideline PDFs
2. Retrieve relevant evidence for each visual question
3. Run a VLM on frame plus retrieved evidence with defer-aware prompting

The current repo is prepared for the `frames_v3` and `questions_v3` setup.

## Project Layout

```text
surg-rag-vqa/
+-- data/
¦   +-- annotations/
¦   ¦   +-- questions_v3.json
¦   ¦   +-- retrieval_eval_v3.json
¦   +-- cholec_raw/
¦   +-- frames_v3/
+-- docs/
¦   +-- raw/
¦   +-- chunks/
+-- results/
+-- scripts/
¦   +-- build_corpus_v2.py
¦   +-- retrieval.py
¦   +-- rag_vqa_pipeline.py
¦   +-- evaluate.py
¦   +-- generate_annotations_v3.py
¦   +-- download_hf_models.py
¦   +-- config.py
+-- .env.example
+-- requirements.txt
```

## Runtime Modes

There are three practical runtime modes:

1. `mock_vlm`
   Use this for smoke testing retrieval and pipeline wiring without loading a VLM.

2. `local_hf`
   Use this on a GPU server with Hugging Face models.

3. `openai`
   Use this only if you intentionally want to call an OpenAI model.

## Local Dev Setup

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Linux server:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## GPU Server Setup

Create a fresh environment on the server. Do not copy the local `venv/` directory.

Typical sequence:

```bash
git clone <your-repo-url>
cd surg-rag-vqa
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

If you need a CUDA-specific PyTorch build, install that first, then install the rest of `requirements.txt`.

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check GPU visibility:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Environment Configuration

Edit `.env` after copying from `.env.example`.

### Safe local smoke-test config

```env
VLM_PROVIDER=mock_vlm
RETRIEVAL_MODE=hybrid
DENSE_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

### Stable server config

This is the most practical starting point for a strong GPU server:

```env
VLM_PROVIDER=local_hf
HF_TOKEN=hf_your_read_token_here
HF_LOCAL_FILES_ONLY=0
HF_CACHE_DIR=
LOCAL_VLM_MODEL=llava-hf/llava-1.5-7b-hf
LOCAL_VLM_MAX_NEW_TOKENS=256
RETRIEVAL_MODE=hybrid
DENSE_MODEL_NAME=BAAI/bge-large-en-v1.5
```

### Optional stronger VLM

If your server stack supports it cleanly, you can later try:

```env
LOCAL_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

## Download Hugging Face Models

Before running the full local Hugging Face pipeline, pre-download the models:

```bash
python scripts/download_hf_models.py
```

After successful download, you can force offline loading:

```bash
export HF_LOCAL_FILES_ONLY=1
```

Windows:

```powershell
$env:HF_LOCAL_FILES_ONLY="1"
```

## Run Order

### 1. Build the document corpus

Run when you change PDFs in `docs/raw/`:

```bash
python scripts/build_corpus_v2.py
```

### 2. Test retrieval only

```bash
python scripts/retrieval.py
```

If dense retrieval is unavailable, you can force BM25-only:

```bash
export RETRIEVAL_MODE=bm25_only
python scripts/retrieval.py
```

### 3. Run the full RAG-VQA pipeline

```bash
python scripts/rag_vqa_pipeline.py
```

Outputs are written to:

- `results/spike_results_v3.json`

### 4. Evaluate results

```bash
python scripts/evaluate.py
```

## What Each Script Does

- `scripts/build_corpus_v2.py`
  Builds chunked knowledge files from guideline PDFs.

- `scripts/retrieval.py`
  Runs sparse or hybrid retrieval over the built corpus.

- `scripts/rag_vqa_pipeline.py`
  Runs the full frame to retrieval to VLM to parse to save pipeline.

- `scripts/download_hf_models.py`
  Pre-downloads retrieval and VLM models for Hugging Face local mode.

- `scripts/generate_annotations_v3.py`
  Regenerates `questions_v3.json` and `retrieval_eval_v3.json`.

## Current Defaults

The code currently points to:

- frames: `data/frames_v3`
- questions: `data/annotations/questions_v3.json`
- retrieval eval: `data/annotations/retrieval_eval_v3.json`
- results: `results/spike_results_v3.json`

## Notes

- Do not commit a real `.env` with secrets.
- If you shared a real Hugging Face token during development, rotate it before public use.
- If local Hugging Face VLM loading fails, switch to `mock_vlm` to validate the rest of the pipeline first.
