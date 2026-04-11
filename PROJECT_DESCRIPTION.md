# PROJECT DESCRIPTION

## 1. One-sentence summary

This project is a safety-aware Surgical Retrieval-Augmented Visual Question Answering (RAG-VQA) pipeline for laparoscopic cholecystectomy that combines laparoscopic frames, guideline-grounded retrieval, and a defer-capable vision-language model to answer only when the visual and textual evidence is strong enough.

## 2. What the project is actually trying to do

The repository is not a generic medical VQA demo and not only a retrieval experiment. The current codebase is trying to answer a more specific research question:

> Can a surgical AI assistant answer clinically motivated questions about a laparoscopic frame more safely when its answer is grounded in retrieved surgical evidence and when it is explicitly allowed to defer?

That research question is reflected in three design decisions that appear consistently across the source code:

1. The system is image-conditioned.
   It starts from a laparoscopic frame rather than a purely text-only question.

2. The system is retrieval-grounded.
   It does not trust the VLM to answer from image priors alone. It first retrieves evidence from curated surgical documents and anatomy references.

3. The system is safety-aware.
   The VLM is instructed to return `DEFER` when anatomy is unclear, the image quality is poor, or the retrieved evidence is insufficient or contradictory.

In other words, the project is exploring whether surgical frame understanding becomes more clinically useful when it is treated as a grounded decision-support problem instead of a plain captioning or pure recognition problem.

## 3. High-level workflow of the current codebase

The code in the repository implements the following end-to-end workflow:

1. A curated subset of surgical frames is stored in `data/frames_v3/`.
2. A blueprint and frame metadata are converted into task-ready annotations in `data/annotations/questions_v3.json` and `data/annotations/retrieval_eval_v3.json`.
3. A curated surgical knowledge corpus is built from PDFs in `docs/raw/`.
4. The corpus builder produces a structured chunk file at `docs/chunks/chunks_v3.jsonl`.
5. The retrieval module indexes those chunks and retrieves parent-expanded evidence for each question.
6. The RAG-VQA pipeline combines:
   - the frame,
   - the question,
   - the retrieved evidence,
   - and a safety-oriented system prompt
   and then calls either a local Hugging Face VLM, an OpenAI model, or a mock mode.
7. The raw model response is parsed into a structured result with fields such as:
   - answer text,
   - confidence,
   - defer flag,
   - retrieved chunk IDs,
   - evidence-card metadata,
   - latency,
   - and error state if the run failed.
8. The evaluation script computes defer-related metrics, confidence distribution, question-type breakdowns, latency, and per-question summaries.

This architecture means the project is not only about obtaining an answer. It is also about:

- evidence provenance,
- traceable retrieval,
- calibrated abstention behavior,
- and structured output for later analysis.

## 4. Current project state: what version is live now

The real working state of the repository is centered on the `v3` setup, not the older `v1` or `v2` experiments.

The key active artifacts are:

- frames: `data/frames_v3/`
- questions: `data/annotations/questions_v3.json`
- retrieval eval set: `data/annotations/retrieval_eval_v3.json`
- retrieval corpus: `docs/chunks/chunks_v3.jsonl`
- main result file: `results/spike_results_v3.json`

The important scripts now are:

- `scripts/build_corpus_v3.py`
- `scripts/generate_annotations_v3.py`
- `scripts/retrieval_v3.py`
- `scripts/rag_vqa_pipeline.py`
- `scripts/evaluate.py`
- `scripts/download_hf_models.py`
- `scripts/config.py`

There are still some legacy scripts in the repo such as `build_corpus.py`, `build_corpus_v2.py`, `retrieval.py`, and some older helper scripts, but the current code path for the main experiment is the `v3` stack above.

## 5. Main folders and their real role in the project

### `data/`

This is the main data workspace for frame-based experiments.

- `data/cholec_raw/`
  Stores the original raw surgical image data source used earlier in the pipeline. This is the raw reservoir from which curated subsets can be generated.

- `data/frames_v3/`
  Stores the current curated frame subset used by the experiment.
  Based on the scripts and surrounding files, this folder is intended to contain:
  - selected frame images,
  - frame metadata,
  - a blueprint describing question type, difficulty, and defer labeling.

- `data/annotations/`
  Stores generated annotations that are used directly by retrieval and VQA:
  - `questions_v3.json`
  - `retrieval_eval_v3.json`

### `docs/`

This is the retrieval knowledge area.

- `docs/raw/`
  Contains the source medical documents that define the retrieval corpus.
  The current `config.py` manifest shows the project is built around these document groups:
  - SAGES Safe Cholecystectomy guideline
  - Tokyo Guidelines 2018 safe steps
  - WSES bile duct injury guideline
  - CVS review
  - Rouviere's sulcus anatomy review
  - CholecSeg8k class definitions
  - WHO surgical safety checklist

- `docs/chunks/`
  Contains prebuilt chunk corpora.
  The active output for the current system is `chunks_v3.jsonl`.

### `scripts/`

This folder contains the implementation of the full pipeline:

- corpus construction
- annotation generation
- retrieval
- VLM execution
- evaluation
- orchestration
- model pre-download

### `results/`

Stores runtime outputs, including:

- `spike_results_v3.json`
- evaluation outputs generated by `evaluate.py`
- mock retrieval outputs when running in mock mode

### `notebooks/`

Used for exploratory inspection and analysis outside the scripted pipeline.

## 6. Configuration system and runtime control

The central runtime configuration is in `scripts/config.py`.

The configuration file does four important jobs:

1. It defines canonical project paths.
   This includes frames, annotations, raw docs, chunk files, and result files.

2. It declares the retrieval corpus manifest.
   Each document has metadata such as:
   - `doc_id`
   - `doc_title`
   - `source_type`
   - `trust_tier`
   - `collection`
   - `priority`
   - `chunk_strategy`
   - `chunk_size`
   - `tags_hint`

3. It defines retrieval defaults.
   These include:
   - `DENSE_MODEL_NAME`
   - `USE_RERANKER`
   - `RERANKER_MODEL_NAME`
   - `RERANK_TOP_N`
   - `HYBRID_ALPHA`
   - `RETRIEVAL_TOP_K`
   - `RETRIEVAL_MODE`
   - `HF_CACHE_DIR`
   - `HF_LOCAL_FILES_ONLY`

4. It defines VLM behavior.
   These include:
   - `VLM_PROVIDER`
   - `OPENAI_VLM_MODEL`
   - `LOCAL_VLM_MODEL`
   - `LOCAL_VLM_MAX_NEW_TOKENS`
   - `VLM_MAX_TOKENS`
   - `VLM_TEMPERATURE`

The current default local model in code is:

```env
LOCAL_VLM_MODEL=llava-hf/llava-1.5-7b-hf
```

This means the present codebase is optimized around a local Hugging Face VLM workflow rather than the older Florence-specific branch that was previously experimented with.

## 7. What `generate_annotations_v3.py` is doing

The annotation generation script is more than a file converter. It turns frame-level metadata into structured VQA tasks.

It reads:

- `data/frames_v3/question_blueprint_v3.json`
- `data/frames_v3/frame_metadata.json`

and writes:

- `data/annotations/questions_v3.json`
- `data/annotations/retrieval_eval_v3.json`

The script currently encodes a specific task ontology with five question types:

1. `recognition`
2. `workflow_phase`
3. `anatomy_landmark`
4. `safety_verification`
5. `risk_pitfall`

For each frame/question item, it constructs:

- a question phrasing,
- a difficulty label,
- a `should_defer` flag,
- a gold-answer stub,
- notes containing upstream scoring context,
- class detections,
- and metadata such as source frame, video ID, and frame index.

This is important because the retrieval and RAG pipeline are not operating on generic free-form VQA. They are operating on a deliberately structured question space that tries to reflect clinically relevant reasoning categories.

The retrieval-eval output also contains:

- relevant keywords,
- a minimum acceptable chunk needle,
- expected collections,
- difficulty,
- and defer status.

That means the eval set is already aligned with the retrieval design, not just with the final VLM answer stage.

## 8. What `build_corpus_v3.py` is doing and why it matters

`build_corpus_v3.py` is one of the most technically important parts of the repository because it determines the quality of the evidence the VLM will see.

This script builds `docs/chunks/chunks_v3.jsonl` and introduces several substantial upgrades over earlier versions.

### 8.1 Core responsibilities

The script:

- reads source documents from `docs/raw/`
- extracts page-level text from PDFs
- cleans noisy extraction artifacts
- detects sections and heading structure
- chunks text into parent and child units
- links children to parents
- extracts semantic tags
- builds contextualized retrieval text
- creates section and document summaries
- validates parent-child integrity
- writes the final v3 corpus

### 8.2 PDF extraction strategy

The script tries `pypdf` first and then falls back to `pdfplumber` if:

- extraction fails,
- the text is too short,
- or the extracted text looks garbled.

This is a practical design choice for medical PDFs because guideline documents often vary in layout quality and encoding.

### 8.3 Mojibake normalization

The corpus builder includes explicit mojibake normalization logic.

This is significant because PDF extraction often produces broken punctuation, quotation marks, or double-encoded characters. The script attempts to normalize those artifacts before chunking and indexing.

### 8.4 Front-matter and junk removal

The text cleaner is not just removing blank lines.
It explicitly tries to remove:

- page numbers
- downloads/copyright boilerplate
- DOI lines
- URL lines
- reference-heavy blocks
- front matter such as author and affiliation blocks
- table-of-contents-like material

This improves retrieval quality because otherwise BM25 and dense retrieval can latch onto useless but frequent terms from article metadata rather than surgical content.

### 8.5 Section detection and heading parsing

The v3 builder tightens heading detection compared with earlier versions.

It uses:

- heading regex patterns,
- heading rejection filters,
- and heading-stack tracking

to build section-level structure and heading paths such as hierarchical section context.

This matters because retrieval quality is improved when the system knows not just the chunk text, but also the section in which that text appeared.

### 8.6 Parent-child hierarchical chunking

One of the key upgrades in v3 is the parent-child chunk hierarchy.

The script creates:

- parent chunks for broader evidence context
- child chunks for finer-grained retrieval units

The token defaults are currently:

- child chunks: 250 tokens
- parent chunks: 800 tokens
- child overlap: 30 tokens
- parent overlap: 80 tokens

Parent-child linking is based on sentence-span overlap rather than naive equal splitting.

This is a strong design choice because it allows:

- precise retrieval on smaller evidence units
- but richer evidence packaging for the final prompt

That same design is later used directly by `retrieval_v3.py`, which retrieves on children and expands to parents for prompt construction.

### 8.7 Tag extraction

The corpus builder extracts multiple tag families from chunk text:

- `anatomy_tags`
- `instrument_tags`
- `action_tags`
- `risk_tags`
- `phase_scope`

It uses word-boundary matching and guarded alias logic to reduce false positives from overly short aliases.

These tags later become important retrieval signals in the field-aware BM25 and prior-boosting logic.

### 8.8 Contextualized retrieval text

Each chunk gets a `contextualized_text` field.

This is one of the most important changes in the current retrieval design.

Instead of retrieving on raw chunk text alone, the retriever can use a richer representation that includes:

- document title
- section title / heading path
- collection context
- selected tags
- and the actual chunk content

This makes the retrieval stage more semantically informed and less brittle.

### 8.9 Summary levels

The script also creates:

- `section_summary` chunks
- `document_summary` chunks

These are useful for future experiments even if the current retrieval system focuses mainly on child-first retrieval.

### 8.10 Validation

The builder ends with a validation pass that checks:

- how many child chunks received `parent_id`
- parent-child text consistency
- obvious tag false positives
- remaining mojibake

This indicates that the corpus builder is not just a preprocessing script. It is already treated as a measurable quality-control stage.

## 9. What `retrieval_v3.py` is doing

`retrieval_v3.py` is the retrieval engine that the current RAG-VQA system actually uses.

The file implements a retrieval design with several layers:

### 9.1 Child-first indexing, parent-expanded evidence

The retriever:

- loads `chunks_v3.jsonl` if present
- indexes mainly `child` chunks for retrieval
- then expands selected matches to their linked `parent` chunks for downstream evidence packaging

This is a key architectural decision because it balances:

- retrieval precision
- evidence completeness
- and prompt readability

### 9.2 Field-aware BM25

The retriever does not treat each chunk as one flat string for BM25.
It builds BM25 indexes over multiple fields:

- `contextualized_text`
- `doc_title`
- `section_title`
- `chunk_type`
- `anatomy_tags`
- `risk_tags`
- `phase_scope`
- `instrument_tags`
- `action_tags`

Each field has an explicit weight.

This means sparse retrieval is already knowledge-structured rather than plain bag-of-words matching.

### 9.3 Dense retrieval

When enabled, the retriever uses a sentence-transformer embedding model over `contextualized_text`.

The current intended dense model from the environment examples is:

```env
DENSE_MODEL_NAME=BAAI/bge-large-en-v1.5
```

Embeddings are normalized and indexed with FAISS for inner-product search.

### 9.4 Optional reranking

If `USE_RERANKER=1`, the retriever loads a reranker model and reranks the top candidate pool.

The current intended reranker is:

```env
RERANKER_MODEL_NAME=BAAI/bge-reranker-large
```

This gives the system a three-stage retrieval stack:

1. sparse candidate scoring
2. dense candidate scoring
3. optional neural reranking

### 9.5 Query conditioning by question type

This is one of the strongest parts of the current project design.

The retriever uses `question_type` hints to adjust:

- preferred collections
- preferred chunk types
- extra query terms

For example:

- `recognition` favors visual ontology and anatomy-landmark content
- `safety_verification` favors safe-chole and complication-management evidence
- `risk_pitfall` favors complication and bailout-oriented evidence

This means retrieval is not generic across tasks. It is explicitly conditioned on the clinical intent of the question.

### 9.6 Query expansion using detected classes

If a frame has `classes_detected`, the retriever maps class labels into additional query terms.

For example:

- `grasper`
- `cystic_duct`
- `cystic_artery`
- `hepatic_vein`
- `liver_ligament`

can be expanded into anatomical or operative terms that help the retriever search for more relevant evidence.

This is an important multimodal bridge in the system:

- vision-side metadata from segmentation/class detection
- influences text-side retrieval behavior

### 9.7 Fusion and priors

The retriever combines sparse and dense rankings via reciprocal rank fusion, then adjusts scores using:

- collection priority
- trust tier
- preferred collection membership
- preferred chunk types
- low-value section penalties
- question-type-specific boosts or penalties

This makes the retrieval stage highly engineered rather than merely off-the-shelf.

### 9.8 Adaptive evidence selection

After ranking, the retriever performs adaptive selection with diversity constraints.

It tries to avoid:

- too many chunks from the same parent
- too much duplication from the same section

This is important because VLM prompts degrade quickly if the top evidence is redundant.

### 9.9 Evidence packaging

The final retrieved objects contain more than raw text.
Each packaged item can include:

- `matched_chunk_id`
- `evidence_chunk_id`
- `matched_level`
- `evidence_level`
- `evidence_text`
- `evidence_raw_text`
- `evidence_card`

This packaging is what makes the RAG pipeline traceable and inspectable.

### 9.10 Current observed state

From our recent workflow, retrieval has already been brought to a strong state and reached full retrieval success on the current evaluation setup.

That means the current bottleneck has likely shifted away from retrieval and more toward:

- VLM answer quality
- defer calibration
- prompt robustness
- and answer evaluation quality

## 10. What `rag_vqa_pipeline.py` is doing now

This is the script that runs the actual frame + retrieval + VLM pipeline.

The current version of the file has already been cleaned to align with the present project direction.

### 10.1 Current pipeline behavior

For each question row in `questions_v3.json`, the script:

1. loads the frame from `data/frames_v3/`
2. retrieves evidence using `retrieval_v3.SurgicalRetriever`
3. passes `question_type` and `classes_detected` into retrieval
4. builds a safety-aware system prompt containing retrieved evidence
5. calls the selected VLM provider
6. parses the raw answer into structured fields
7. writes the result to the main results file

### 10.2 Evidence formatting

The prompt includes labeled evidence blocks with metadata such as:

- document title
- retrieval score
- collection
- chunk type
- section title

This is important for transparency and potentially improves the VLM's ability to distinguish high-level evidence sources.

### 10.3 VLM modes

The pipeline currently supports three runtime modes:

- `mock_vlm`
- `openai`
- `local_hf`

The local Hugging Face branch is now cleaner and aligned with the current intended workflow.
Earlier Florence-specific logic was removed from this main pipeline because it was no longer part of the active experiment path.

### 10.4 Local HF behavior

The local HF path:

- loads `AutoProcessor`
- loads `AutoModelForImageTextToText`
- uses the configured Hugging Face cache and token settings
- pushes tensors to GPU when available
- and generates deterministic outputs with `do_sample=False`

### 10.5 Response parsing

The response parser extracts:

- raw response text
- defer flag
- confidence label
- parsed answer

It now handles formatting more robustly than before and is less brittle when the model returns slightly different casing.

### 10.6 Error handling

The current version also fixes an earlier logic issue where runtime exceptions could be stored without a proper `error` field and then be counted incorrectly in the run summary.

Now, failed items are explicitly marked as errors and are counted correctly in the final summary.

### 10.7 Retrieval trace in output

The result objects now include not only `retrieved_chunks`, but also:

- `retrieved_matched_chunks`
- `retrieved_evidence_chunks`
- `retrieved_scores`
- `retrieved_previews`
- `retrieved_evidence_cards`

This is extremely helpful for later debugging because retrieval in v3 is hierarchical: the matched child chunk is not always the same as the packaged parent evidence chunk.

## 11. What `evaluate.py` is doing

The evaluation script is designed mainly around defer-aware analysis rather than conventional answer-only accuracy.

It computes:

- total valid results
- answered vs deferred counts
- defer TP / FP / FN / TN
- defer precision
- defer recall
- defer F1
- confidence distribution
- breakdown by question type
- breakdown by difficulty
- average latency
- per-question summary rows

This reflects the project's actual objective well:

- a missed defer is treated as dangerous
- an unnecessary defer is treated as overly cautious
- and defer quality is measured explicitly

That is a good fit for surgical decision-support research, where abstention behavior matters as much as answer generation.

## 12. What `download_hf_models.py` is doing

This utility script pre-downloads the models required for local HF execution:

- dense retrieval model
- optional reranker model
- local VLM

Its purpose is to let the server cache models before the main pipeline runs.

This matters operationally because the project uses several large Hugging Face assets, and downloading them lazily during the first full run is slower and harder to debug.

The script respects:

- `HF_CACHE_DIR`
- `HF_TOKEN`
- current retrieval model settings
- current local VLM setting

Operationally, this script is part of the deployment workflow, not just a convenience helper.

## 13. What we have effectively built so far

Looking at the repository as a whole, the work completed so far is much more than wiring together a VLM and some PDFs.

We have effectively built:

### 13.1 A task formulation layer

The project has a defined ontology of question types, difficulty, and defer expectation.

### 13.2 A curated retrieval corpus

The system does not retrieve from the open internet or arbitrary documents.
It retrieves from a deliberately selected surgical knowledge set with trust tiers and document collections.

### 13.3 A structured retrieval engine

The retriever is:

- field-aware
- query-conditioned
- hierarchy-aware
- evidence-packaging aware
- and optimized for clinically themed question types

### 13.4 A grounded multimodal answer pipeline

The VLM is not answering from image alone.
It is answering with access to retrieved surgical evidence and with explicit safety instructions.

### 13.5 A defer-aware evaluation mindset

The project already treats abstention as a first-class target rather than an afterthought.

That is a meaningful research framing choice, especially for surgical support.

## 14. Strengths of the current system

Based on the current source code and recent progress, the main strengths of the project are:

### 14.1 Retrieval is strongly engineered

The retrieval stack is not naive.
It includes:

- document curation
- section structure
- parent-child chunking
- tag extraction
- contextualized text
- query-type priors
- class-conditioned expansion
- reranking
- and adaptive evidence selection

That is likely one of the strongest parts of the project right now.

### 14.2 The project is clinically framed rather than technically generic

Question types such as `safety_verification` and `risk_pitfall` are much closer to real surgical-support reasoning than generic VQA labels.

### 14.3 The defer mechanism is baked into the pipeline design

The model is explicitly instructed to prefer safety over forced answers.

### 14.4 The outputs are inspectable

Because the pipeline stores evidence cards, chunk IDs, and parsed responses, it is possible to audit why a result happened.

## 15. Current limitations visible from the codebase

Even though the project is in a good state, the current codebase still exposes some meaningful limitations.

### 15.1 Evaluation still emphasizes defer metrics more than semantic answer correctness

The evaluation script is good for abstention analysis, but it does not yet perform robust semantic matching between predicted answers and gold answers.

That means the current system is better at answering:

- "Did the model defer correctly?"

than:

- "Was the content of the non-defer answer clinically correct?"

### 15.2 Gold answers are still scaffold-like in many places

The annotation generator creates strong structured stubs, but many `gold_answer` values still look like draft supervision rather than expert-finalized labels.

This is acceptable for a feasibility spike, but it limits how strong final answer-evaluation claims can be.

### 15.3 Prompting is still fairly generic once evidence is inserted

The current RAG prompt is safety-aware, but there is still room to tailor prompting more specifically by question type.

For example:

- recognition questions might need shorter visual identification prompts
- safety verification questions might need stricter criteria-oriented prompts
- risk questions might need stronger hazard-focused prompting

### 15.4 The deployment layer is still partly experimental

The main v3 pipeline is aligned, but some supporting scripts and text in the repo still contain traces of earlier versions or assumptions from prior experiments.

This does not block the current work, but it means the repository still benefits from gradual cleanup and harmonization.

## 16. What the project is most ready for next

Because retrieval is now strong, the most promising next optimization directions are likely:

### 16.1 Improve answer quality after retrieval

The next gains may come from:

- prompt design
- evidence compression
- evidence ordering
- and model choice

### 16.2 Improve defer calibration

The project already supports defer behavior, so now the natural next question is:

> When is the system deferring too little, and when is it deferring too much?

That suggests work on:

- thresholding
- prompt constraints
- question-type-specific defer rules
- uncertainty heuristics from visual quality or retrieval quality

### 16.3 Strengthen answer-side evaluation

The repository would benefit from more systematic scoring for:

- answer correctness
- groundedness to retrieved evidence
- hallucination tendency
- and per-question-type failure modes

### 16.4 Tighten repo consistency

Now that the main path is clearly v3, a good cleanup direction is to make all helper scripts, reports, and documentation fully consistent with:

- `chunks_v3`
- `questions_v3`
- `retrieval_v3`
- `spike_results_v3`
- and the local Hugging Face deployment path

## 17. Practical summary of what we have done together

At this point, the project has evolved into a coherent surgical RAG-VQA prototype with:

- a structured `v3` frame/annotation setup
- a substantially upgraded corpus builder
- a much stronger retrieval engine
- a cleaned local-HF RAG pipeline
- explicit defer-aware outputs
- and an evaluation path designed for feasibility analysis

In practical terms, the current codebase now supports this full experiment:

1. prepare frame metadata and question blueprint
2. generate `questions_v3` and `retrieval_eval_v3`
3. build `chunks_v3` from curated surgical PDFs
4. retrieve evidence with `retrieval_v3`
5. answer with a local Hugging Face VLM or mock/openai mode
6. parse answer vs defer behavior
7. evaluate the run with explicit safety-oriented metrics

That is already a strong foundation for the next stage of optimization.

## 18. Final interpretation

The current repository represents a transition point in the project:

- retrieval is no longer the weakest link
- the pipeline structure is much more mature
- and the next meaningful gains will likely come from improving answer generation, calibration, and evaluation rigor

So the project is now in a good position to move from:

> "Can we make a safety-aware surgical RAG-VQA pipeline work at all?"

toward:

> "How do we make it more reliable, more interpretable, and more clinically convincing?"
