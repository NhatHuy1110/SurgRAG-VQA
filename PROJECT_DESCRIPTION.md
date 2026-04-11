# PROJECT DESCRIPTION

## 1. Project in one sentence

This project explores a safety-aware Retrieval-Augmented Visual Question Answering (RAG-VQA) pipeline for laparoscopic cholecystectomy: given a surgical frame, the system retrieves relevant guideline evidence, asks a vision-language model to answer a question about the frame, and allows the model to explicitly defer when the image or evidence is not reliable enough for a safe answer.

## 2. What the project is trying to achieve

The repository is not just a generic VQA demo. It is a feasibility spike for a surgical AI workflow with three tightly connected goals:

1. Ground image-based reasoning in surgical knowledge rather than relying on pure visual pattern matching.
2. Support clinically relevant question types such as anatomy landmarks, safety verification, workflow phase, and risk/pitfall recognition.
3. Add a safety-first `DEFER` mechanism so the system can refuse to answer when anatomy is unclear, visibility is poor, or retrieved evidence is insufficient.

In practical terms, the project is testing whether a surgical assistant can combine:

- visual input from laparoscopic frames,
- textual evidence from surgical guidelines and anatomy references,
- and a structured answer/defer policy

to produce safer answers than a plain image-only model.

## 3. The big picture workflow

The codebase is organized around the following end-to-end workflow:

1. Raw laparoscopic frames and masks are collected from the CholecSeg8k-style source data in `data/cholec_raw/`.
2. A frame selection pipeline chooses a curated subset of frames, assigns difficulty, proposes question types, and marks some frames as should-defer cases.
3. Annotation generation scripts turn that frame blueprint into question files and retrieval-evaluation files.
4. A corpus builder parses guideline PDFs in `docs/raw/`, cleans the text, chunks it, and writes a structured retrieval corpus to `docs/chunks/chunks_v2.jsonl`.
5. A retrieval module uses BM25 or hybrid BM25+dense retrieval to find the most relevant evidence chunks for each question.
6. A VLM pipeline combines the frame, the question, and the retrieved evidence into a prompt and asks either:
   - an OpenAI vision model,
   - a local Hugging Face VLM,
   - or a mock mode with no VLM call.
7. The raw answer is parsed into a structured format with:
   - answer text,
   - confidence,
   - defer flag.
8. An evaluation script summarizes defer behavior, answer/defer counts, latency, and per-question outcomes.

## 4. Core problem formulation

The working task is effectively:

> Given a laparoscopic image frame and a clinically motivated question, retrieve relevant surgical knowledge and answer only when it is safe to do so.

This makes the project different from standard medical VQA in two important ways:

- It is retrieval-augmented, so textual evidence is part of the reasoning chain.
- It is defer-aware, so abstention is treated as a first-class behavior instead of a failure mode.

## 5. Main folders and what they contain

### `data/`

This is the main data area. It contains both raw and processed image-related assets.

- `data/cholec_raw/`
  Raw source laparoscopic data, organized by video and frame folders. The files include original endoscopic frames and mask variants such as color masks and watershed masks.

- `data/frames_v3/`
  The curated frame subset currently used by the project. This folder contains:
  - `frame_001.png` through `frame_100.png`
  - corresponding `_mask` files
  - `frame_metadata.json`
  - `question_blueprint_v3.json`
  - `selection_summary.json`
  - `validation_report.json`

- `data/annotations/`
  Generated task annotations used by the pipeline:
  - `questions_v3.json`
  - `retrieval_eval_v3.json`
  There are also older `v1` files kept for previous experiments.

### `docs/`

This folder contains the knowledge source used for retrieval.

- `docs/raw/`
  PDF documents such as:
  - SAGES safe cholecystectomy guideline
  - Tokyo Guidelines 2018 safe steps
  - WSES bile duct injury guideline
  - CVS review
  - Rouviere's sulcus article
  - WHO surgical safety checklist
  - CholecSeg8k class definitions

- `docs/chunks/`
  Prebuilt chunk files:
  - `chunks_v1.jsonl`
  - `chunks_v2.jsonl`

These chunks are the retrieval corpus consumed by the retriever.

### `scripts/`

This is the main implementation folder. It contains the frame selection logic, corpus building, retrieval, pipeline execution, model download utilities, and evaluation.

### `results/`

This contains pipeline outputs. Currently the main file is:

- `results/spike_results_v3.json`

The folder is intended to also hold evaluation outputs such as markdown reports and metrics JSON.

### `notebooks/`

This currently contains `spike_analysis.ipynb`, likely used for exploratory analysis or inspection during development.

## 6. Current experimental dataset state

The current project state is centered on the `v3` setup:

- frames: `data/frames_v3/`
- questions: `data/annotations/questions_v3.json`
- retrieval eval: `data/annotations/retrieval_eval_v3.json`
- output results: `results/spike_results_v3.json`

The selected frame set is already balanced according to `data/frames_v3/selection_summary.json`:

- 100 selected frames total
- 15 recognition questions
- 20 workflow phase questions
- 25 anatomy landmark questions
- 25 safety verification questions
- 15 risk/pitfall questions
- 30 easy, 40 medium, 30 hard cases
- 25 should-defer cases
- frames drawn from 17 videos

This means the repository already contains a reasonably structured benchmark-like subset rather than a random sample of images.

## 7. How frame selection works

The frame selection pipeline is implemented in the large script currently stored as `scripts/frames_selection_v2.py`, although the script header describes it as a `v3` selector.

This script is one of the most important pieces in the repo because it creates the foundation for the downstream evaluation set.

### What it does

It scans raw frame folders and uses both image content and mask-derived semantics to build candidate metadata for each frame.

### Key ideas in the selector

- It reads watershed masks instead of guessing class colors.
- It maps segmentation IDs to semantic classes such as liver, gallbladder, cystic duct, grasper, blood, and hepatic vein.
- It computes visual quality signals such as brightness, contrast, sharpness, edge density, and specular reflection.
- It infers frame difficulty (`easy`, `medium`, `hard`).
- It scores each frame against target question types:
  - `recognition`
  - `workflow_phase`
  - `anatomy_landmark`
  - `safety_verification`
  - `risk_pitfall`
- It estimates whether the frame should be a defer case.
- It enforces diversity constraints:
  - max frames per video,
  - minimum temporal gap within the same video,
  - near-duplicate filtering using perceptual hash.
- It saves:
  - copied frame images,
  - copied masks,
  - metadata,
  - a question blueprint,
  - selection summary,
  - validation report.

### Why this matters

The project is not manually annotating from scratch. Instead, it creates a structured scaffold that mixes automatic frame mining with human-review-ready question drafting. That makes the project closer to a scalable dataset-building pipeline than a one-off demo.

## 8. How question and retrieval annotations are generated

The script `scripts/generate_annotations_v3.py` reads:

- `data/frames_v3/question_blueprint_v3.json`
- `data/frames_v3/frame_metadata.json`

and produces:

- `data/annotations/questions_v3.json`
- `data/annotations/retrieval_eval_v3.json`

### `questions_v3.json`

Each question entry includes:

- `qid`
- selected frame filename
- natural-language question
- question type
- difficulty
- should-defer flag
- draft gold answer
- notes for manual review
- source frame metadata
- detected classes

The gold answers are currently scaffolded, not definitive expert-validated labels. The script explicitly marks annotations as:

`ready_run_needs_expert_review`

That is an important clue about project maturity: the benchmark is semi-automatic and still expects expert review.

### `retrieval_eval_v3.json`

This file is used to test retrieval quality. For each question, it stores:

- the question text,
- a small set of relevant keywords,
- a minimum acceptable substring that should appear in a retrieved chunk,
- expected collections,
- question type and difficulty,
- annotation status.

So retrieval evaluation is currently lightweight and heuristic, based on containment checks rather than deep human relevance judgments.

## 9. How the text corpus is built

The main corpus builder is `scripts/build_corpus_v2.py`.

### Inputs

It reads raw PDF or text files from `docs/raw/`.

### Processing stages

1. Extract text using `pypdf`, with fallback to `pdfplumber` when extraction quality is poor.
2. Clean the text aggressively:
   - remove junk lines,
   - remove page numbers and headers/footers,
   - remove copyright and URL noise,
   - truncate at references sections.
3. Detect sections and split content semantically.
4. Apply source-specific chunking strategies such as:
   - `section_aware`
   - `paragraph`
   - `step_based`
   - `lexicon`
5. Enrich each chunk with metadata:
   - document ID and title
   - source type
   - trust tier
   - collection
   - section title
   - chunk type
   - anatomy/action/instrument/risk tags
   - operative phase scope
   - character length
6. Write one JSON object per line into `docs/chunks/chunks_v2.jsonl`.

### Knowledge design

The corpus is not treated as a flat bag of PDFs. `scripts/config.py` defines a document manifest with domain-specific structure:

- core safe cholecystectomy guidelines
- complication management guidelines
- anatomy and landmark reviews
- visual ontology references
- general surgical safety checklist material

Each document has metadata such as:

- `trust_tier`
- `collection`
- `priority`
- `chunk_strategy`
- `tags_hint`

This shows that the project is trying to encode source quality and surgical role directly into retrieval.

## 10. How retrieval works

Retrieval is implemented in `scripts/retrieval.py` through `SurgicalRetrieverV2`.

### Retrieval modes

- `bm25_only`
- `hybrid`

### Retrieval pipeline

1. Load all chunks from the JSONL corpus.
2. Build a BM25 index over chunk text.
3. Optionally build a dense index using:
   - `sentence-transformers`
   - FAISS
4. Retrieve candidates with sparse and dense search.
5. Normalize and combine scores using a weighted hybrid formula.
6. Apply collection-aware boosting using priorities from `config.py`.
7. Optionally filter by collection or chunk type.

### Why this is interesting

Retrieval is not generic document search. It is domain-shaped:

- chunk metadata is used downstream,
- high-value collections are boosted,
- retrieval can be evaluated against question-specific expectations,
- and the system supports failure fallback to BM25-only when dense retrieval is unavailable.

## 11. How the RAG-VQA pipeline works

The main runtime pipeline is `scripts/rag_vqa_pipeline.py`.

### Inputs

- frame image from `data/frames_v3/`
- question from `questions_v3.json`
- retrieved chunks from the retriever

### Prompting strategy

The pipeline builds a system prompt that:

- frames the assistant as a surgical AI assistant,
- includes retrieved evidence as context,
- enforces a safety-critical `DEFER` rule,
- requires one of two output formats:
  - `ANSWER: ... | CONFIDENCE: ...`
  - `DEFER: ...`

This is a key project design choice: the answer format is intentionally constrained so the output can be parsed and evaluated consistently.

### Supported VLM backends

- `mock_vlm`
  For smoke testing retrieval and pipeline wiring without any VLM call.

- `openai`
  Sends the frame and prompt to an OpenAI-compatible vision model.

- `local_hf`
  Runs a Hugging Face vision-language model locally.

### Local HF model support

The code is designed to support models such as:

- `llava-hf/llava-1.5-7b-hf`
- optionally Florence-style models

The pipeline loads the processor and model once, caches them in global variables, and then reuses them across questions.

### Batch run behavior

For each question, the script:

1. resolves the frame path,
2. retrieves top-k chunks,
3. builds the prompt,
4. calls the VLM,
5. parses the answer,
6. attaches metadata such as:
   - gold answer,
   - should-defer flag,
   - question type,
   - difficulty,
   - latency.

It then writes all results to `results/spike_results_v3.json`.

## 12. How evaluation works

The script `scripts/evaluate.py` evaluates the saved results.

Its focus is not traditional exact-match VQA accuracy. Instead, it emphasizes the defer mechanism.

### Metrics computed

- total answered vs deferred
- defer true positives
- defer false positives
- defer false negatives
- defer true negatives
- defer precision, recall, F1
- confidence distribution
- per-question-type breakdown
- per-difficulty breakdown
- average latency

### Generated outputs

The script is designed to produce:

- console report
- `results/evaluation_report.md`
- `results/metrics.json`

### Why this evaluation is important

This project is explicitly safety-oriented. A missed defer on an unsafe frame is treated as a dangerous failure mode. That makes the evaluation closer to risk-sensitive abstention analysis than ordinary VQA benchmarking.

## 13. Current repository status and maturity

The repository is already beyond the idea stage. It contains:

- a curated frame subset,
- generated questions and retrieval evaluation files,
- a structured surgical text corpus,
- a retriever,
- a VLM integration pipeline,
- and an evaluation script.

However, it is still clearly a research prototype / spike rather than a production-ready system.

### Signs of maturity

- the workflow is end-to-end,
- the frame subset is balanced,
- corpus metadata is rich,
- retrieval is configurable,
- and the defer mechanism is integrated into both prompting and evaluation.

### Signs it is still experimental

- gold answers are scaffolded and marked as needing expert review,
- some scripts are older or out of sync with the current `v3` setup,
- the orchestration script still references older file names and older corpus logic in places,
- current result files indicate at least one runtime issue with the local Hugging Face path.

## 14. Important inconsistencies and technical debt already visible

Anyone reading this repo should know that not every script is equally current.

### A. `v3` logic stored in a `v2` filename

The file named `scripts/frames_selection_v2.py` has a header describing itself as `frames_selection_v3.py`. This suggests the script evolved but was not renamed consistently.

### B. Old scripts still coexist

The repo still contains older versions such as:

- `scripts/build_corpus.py`
- `scripts/frames_selection.py`

These appear to reflect earlier iterations and should not be assumed to match current config or outputs.

### C. `run_all.py` is not fully aligned with current naming

`scripts/run_all.py` still references older output names such as `spike_results_v1.json` in comments and uses `build_corpus.py`, while the rest of the repo has moved to the `v3` setup and `build_corpus_v2.py`.

### D. Current saved results indicate a model/runtime issue

`results/spike_results_v3.json` currently shows repeated errors of the form:

`cannot import name 'AutoModelForVision2Seq' from 'transformers'`

So the saved `v3` result file appears to reflect a failed local-HF run rather than a successful full evaluation pass.

This does not invalidate the project structure, but it does mean the present repo state is "pipeline mostly assembled, but current local model execution still needs fixing."

## 15. External dependencies and runtime assumptions

The project expects a Python environment with packages from `requirements.txt`, including:

- PyTorch
- transformers
- sentence-transformers
- FAISS
- BM25
- PDF parsing libraries
- Pillow / numpy
- OpenAI SDK

It also assumes one of the following runtime contexts:

- local smoke test on CPU using `mock_vlm`,
- GPU server for local Hugging Face VLM,
- or cloud/API usage for OpenAI vision models.

Environment variables are configured through `.env` using `.env.example`.

## 16. What the project is really about at a research level

At a research level, this project is trying to answer:

> Can we build a surgical assistant that does not just recognize what is in an image, but uses guideline-based evidence and knows when not to answer?

That makes the project sit at the intersection of:

- medical VQA,
- retrieval-augmented generation,
- surgical scene understanding,
- risk-aware abstention / defer mechanisms,
- and semi-automatic dataset construction.

## 17. If someone new joins this project, what should they understand first

A new collaborator should understand the project in this order:

1. `data/frames_v3/` is the curated benchmark subset.
2. `questions_v3.json` is generated, not purely hand-authored.
3. `docs/raw/` plus `build_corpus_v2.py` define the knowledge base.
4. `retrieval.py` is the bridge between question text and evidence.
5. `rag_vqa_pipeline.py` is the main end-to-end experiment runner.
6. `evaluate.py` judges not only answers, but also whether the model deferred appropriately.
7. The project is currently strongest as a research prototype for pipeline design, not yet as a finalized benchmark or polished deployable system.

## 18. Recommended mental model of the repository

The cleanest way to think about this repository is:

- `frames_selection_*` builds the image benchmark,
- `generate_annotations_v3.py` builds the task files,
- `build_corpus_v2.py` builds the knowledge base,
- `retrieval.py` finds evidence,
- `rag_vqa_pipeline.py` answers or defers,
- `evaluate.py` measures safety-oriented behavior.

## 19. Final summary

This repository is a surgical RAG-VQA research prototype for laparoscopic cholecystectomy. It combines curated laparoscopic frames, guideline-derived retrieval corpora, hybrid retrieval, and a vision-language model with a built-in defer policy. The project is already structured enough to support meaningful experiments, especially around safety-aware answering and abstention, but it still contains annotation scaffolds, version drift between scripts, and at least one unresolved runtime issue in the current local-HF execution path.
