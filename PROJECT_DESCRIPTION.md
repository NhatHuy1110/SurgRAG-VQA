# PROJECT DESCRIPTION - SurgRAG-VQA

## 1. Muc tieu cua du an

`SurgRAG-VQA` la mot du an thu nghiem theo huong `Retrieval-Augmented Vision-Language Question Answering` cho boi canh `laparoscopic cholecystectomy` (cat tui mat noi soi).

Muc tieu trung tam cua he thong la:

1. Nhan dau vao la mot frame noi soi trong qua trinh mo.
2. Dat ra mot cau hoi lien quan den frame do.
3. Truy hoi tri thuc phau thuat tu tap tai lieu chuyen mon.
4. Ket hop `image evidence` + `retrieved knowledge` de tra loi.
5. Neu thong tin khong du an toan thi he thong phai biet `defer` thay vi tra loi lieu.

Noi cach khac, day khong phai la mot he thong image captioning thong thuong, ma la mot pipeline danh cho `clinical reasoning under uncertainty`, trong do:

- `Vision` dung de nhin frame.
- `Retrieval` dung de bo sung ngu canh hoc thuat.
- `Safety logic` dung de tranh tra loi sai trong cac tinh huong nguy hiem.

## 2. Bai toan ma chung ta dang giai

Trong phau thuat noi soi, mot frame don le thuong:

- chi cho thay mot phan nho cua truong mo,
- co the mo, toi, nhieu khoi, mau, hay goc nhin xau,
- khong phai luc nao cung du de khang dinh anatomy hoac phase,
- va mot cau tra loi sai co the dan den suy luan nguy hiem.

Vi vay, bai toan cua chung ta khong chi la "model nhin thay gi?" ma la:

1. Frame nay co du thong tin de tra loi an toan khong?
2. Neu co, cau tra loi nao la hop ly nhat?
3. Neu khong, he thong co biet `defer` dung luc khong?

Do la ly do du an duoc thiet ke xoay quanh 5 nhom cau hoi:

- `recognition`
- `workflow_phase`
- `anatomy_landmark`
- `safety_verification`
- `risk_pitfall`

## 3. Tong quan kien truc he thong

Flow tong the cua du an duoc thiet ke theo chuoi sau:

1. `Frame selection`
   Chon 100 frame dai dien tu du lieu goc CholecSeg8k, co phan bo theo do kho, loai cau hoi va ti le defer mong muon.

2. `Annotation generation`
   Sinh scaffold cau hoi, gold answer stub va metadata retrieval cho tung frame da chon.

3. `Corpus building`
   Xu ly PDF guideline/review/ontology, lam sach text, cat chunk, gan tag anatomy-risk-phase, tao corpus retrieval.

4. `Retrieval`
   Khi co cau hoi, he thong truy hoi evidence lien quan tu corpus bang BM25 + dense retrieval + reranking.

5. `RAG-VQA inference`
   VLM nhan frame + retrieved evidence + question va tra ve:
   - `ANSWER: ... | CONFIDENCE: ...`
   - hoac `DEFER: ...`

6. `Evaluation`
   Danh gia ket qua bang 3 tang:
   - deterministic metrics,
   - overlap metrics,
   - VLM-as-Judge.

## 4. Cau truc thu muc hien tai

### 4.1. Thu muc goc

- `data/`
  Chua frame, annotation, va du lieu goc CholecSeg8k da tach.

- `docs/`
  Chua tai lieu phau thuat dung lam knowledge base cho retrieval.

- `results/`
  Chua ket qua pipeline inference.

- `scripts/`
  Chua toan bo code chinh cua pipeline.

- `notebooks/`
  Thu muc notebooks phu tro.

- `venv/`
  Moi truong Python cuc bo.

- `requirements.txt`
  Danh sach dependency can cho du an.

- `README.md`
  Tai lieu huong dan nhanh.

- `PROJECT_DESCRIPTION.md`
  Tai lieu mo ta chi tiet du an. File nay da duoc viet lai tu dau.

### 4.2. Data assets hien dang ton tai tren local

#### `data/cholec_raw/`

Day la bo du lieu frame goc theo video. Moi video co nhieu folder con, moi folder thuong tuong ung mot moc frame, ben trong chua:

- frame noi soi,
- watershed mask,
- color mask.

Du lieu nay la nguon dau vao cho `frames_selection.py`.

#### `data/frames/`

Day la bo frame da duoc chon cua mot phien ban cu hon, hien dang co:

- `100` anh frame
- `100` mask tuong ung
- `frame_metadata.json`
- `question_blueprint.json`
- `selection_summary.json`
- `validation_report.json`

No cho thay mot pipeline selection/annotation da tung duoc chay va co artifact thuc.

#### `data/frames_v3/`

Thu muc nay la dich den ma code `v3` hien tai dang ky vong de su dung, nhung tren local hien tai chua co artifact day du.

#### `data/annotations/`

Tren local hien dang co:

- `questions.json`
- `retrieval_eval.json`

Trong khi code `v3` hien tai dang ky vong:

- `questions_v3.json`
- `retrieval_eval_v3.json`

Dieu nay cho thay code va artifact hien tai chua dong bo hoan toan theo mot naming convention duy nhat.

### 4.3. Knowledge assets hien dang ton tai

#### `docs/raw/`

Bo tri thuc retrieval hien tai gom 7 PDF:

1. `sages_safe_chole.pdf`
2. `tokyo_guidelines_2018_safe_steps.pdf`
3. `wses_2020_bdi_guideline.pdf`
4. `cvs_review.pdf`
5. `rouviere_sulcus.pdf`
6. `cholecseg8k_classes.pdf`
7. `who_surgical_checklist.pdf`

Day la tap tai lieu da duoc chon de bao phu:

- safe steps trong lap chole,
- injury prevention,
- bailout strategy,
- anatomy landmarks,
- visual ontology,
- va mot phan safety checklist tong quat.

#### `docs/chunks/`

Tren local hien co:

- `chunks.jsonl`

So luong chunk thuc te hien co trong file nay la `1713`.

Moi dong trong `chunks.jsonl` la mot JSON chunk, co cac truong tieu bieu:

- `chunk_id`
- `doc_id`
- `doc_title`
- `source_type`
- `trust_tier`
- `collection`
- `priority`
- `section_title`
- `section_id`
- `heading_path`
- `page_start`, `page_end`
- `level`
- `chunk_type`
- `anatomy_tags`
- `instrument_tags`
- `action_tags`
- `risk_tags`
- `phase_scope`
- `text`
- `token_count`
- `contextualized_text`
- `child_ids`

Code `v3` hien tai dang huong toi `chunks_v3.jsonl`, nhung artifact local van dang la `chunks.jsonl`.

### 4.4. Ket qua inference hien dang ton tai

Thu muc `results/` hien dang co:

- `spike_results_v3.json`

Day la artifact ket qua inference gan nhat hien co tren local.

Thong ke nhanh tren file nay:

- Tong so sample: `100`
- So sample he thong du doan `defer`: `0`
- Phan bo confidence:
  - `high`: 12
  - `medium`: 2
  - `low`: 2
  - `unknown`: 84
- So response rong (`raw_response` empty): `30`
- So chunk retrieval trung binh moi cau hoi: `4.72`

Thong ke nay rat quan trong vi no cho thay:

- pipeline da chay va sinh duoc ket qua,
- nhung quality cua output VLM trong dot inference do con rat yeu,
- dac biet o kha nang follow format va defer.

## 5. Cac script chinh va vai tro cua tung script

Toan bo logic chinh cua du an nam trong thu muc `scripts/`.

### 5.1. `scripts/config.py`

Day la file cau hinh trung tam cua he thong.

No dinh nghia:

- duong dan cac thu muc/chinh file,
- danh sach tai lieu retrieval (`RAG_DOCUMENTS`),
- chunking config,
- retrieval config,
- VLM config,
- evaluation config.

Nhung gia tri quan trong hien tai:

- `FRAMES_DIR = data/frames_v3`
- `QUESTIONS_FILE = data/annotations/questions_v3.json`
- `RETRIEVAL_EVAL_FILE = data/annotations/retrieval_eval_v3.json`
- `CHUNKS_FILE = docs/chunks/chunks_v3.jsonl`
- `RESULTS_FILE = results/spike_results_v3.json`

Ve model:

- Dense retrieval model mac dinh: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker mac dinh: `BAAI/bge-reranker-large`
- OpenAI VLM mac dinh: `gpt-4o`
- Local VLM mac dinh: `llava-hf/llava-1.5-7b-hf`
- Judge model mac dinh: `Qwen/Qwen2.5-VL-7B-Instruct`

Y nghia:

- `config.py` la noi quyet dinh "pipeline mong cho artifact o dau" va "he thong dang duoc cau hinh theo mode nao".
- Day cung la noi the hien ro nhat rang code hien dang theo naming convention `v3`.

### 5.2. `scripts/frames_selection.py`

Day la script chon `100` frame tu bo du lieu goc de xay bo benchmark VQA.

Tu tuong thiet ke:

1. Di qua `data/cholec_raw/`.
2. Doc `watershed mask` thay vi doan class bang color mask.
3. Trich xuat class semantic tren moi frame.
4. Tinh cac feature:
   - quality score,
   - do sang,
   - contrast,
   - sharpness,
   - edge density,
   - semantic coverage.
5. Uoc luong:
   - `difficulty`
   - `question_type`
   - `defer likelihood`
6. Chon frame bang bucket strategy de dat phan bo muc tieu.

Phan bo muc tieu hien duoc encode trong script:

- `recognition = 15`
- `workflow_phase = 20`
- `anatomy_landmark = 25`
- `safety_verification = 25`
- `risk_pitfall = 15`

Do kho muc tieu:

- `easy = 30`
- `medium = 40`
- `hard = 30`

So frame can `defer` muc tieu:

- `DEFER_TARGET = 25`

Script nay sinh ra cac artifact quan trong:

- frame da copy sang thu muc output,
- `frame_metadata.json`
- `question_blueprint.json`
- `selection_summary.json`
- `validation_report.json`

Luu y:

- Code `v3` hien tai ghi vao `data/frames_v3`.
- Artifact local cu hien dang nam o `data/frames`.

### 5.3. `scripts/generate_annotations.py`

Day la script chuyen `frame_metadata + question_blueprint` thanh bo annotation co cau truc cho pipeline.

Vai tro chinh:

1. Doc `question_blueprint.json`
2. Doc `frame_metadata.json`
3. Sinh cau hoi theo `question_type`
4. Sinh `gold_answer_stub`
5. Sinh `retrieval keywords`
6. Xuat:
   - `questions_v3.json`
   - `retrieval_eval_v3.json`

Noi dung annotation gom:

- `qid`
- `frame`
- `question`
- `question_type`
- `difficulty`
- `should_defer`
- `gold_answer`
- `notes`
- `annotation_status`
- `source_frame`
- `video_id`
- `frame_index`
- `classes_detected`

Y nghia:

- Script nay la cau noi giua `frame benchmark` va `RAG-VQA inference`.
- Day la noi ma benchmark duoc `dong goi` thanh bo cau hoi co the chay pipeline.

Local hien dang co bo cu:

- `questions.json`
- `retrieval_eval.json`

Trong khi script hien tai se tao bo moi:

- `questions_v3.json`
- `retrieval_eval_v3.json`

### 5.4. `scripts/build_corpus.py`

Day la script xay knowledge base retrieval tu bo PDF phau thuat.

Day la mot trong nhung thanh phan quan trong nhat cua du an.

Nhung gi script nay lam:

1. Doc PDF bang `pypdf` hoac `pdfplumber`.
2. Lam sach text:
   - bo junk lines,
   - bo front matter,
   - cat references,
   - chuan hoa mojibake,
   - bo noise do OCR/encoding.
3. Detect section va heading.
4. Chia thanh `parent chunks` va `child chunks`.
5. Tao:
   - `section summaries`
   - `document summary`
6. Gan metadata retrieval:
   - `chunk_type`
   - `collection`
   - `trust_tier`
   - tags anatomy/instrument/action/risk/phase
7. Tao `contextualized_text` cho retrieval.
8. Validate chunk graph.

Script su dung manifest tai lieu tu `config.py`, trong do moi document co:

- `doc_id`
- `doc_title`
- `source_type`
- `trust_tier`
- `collection`
- `priority`
- `chunk_strategy`
- `chunk_size`
- `tags_hint`

Muc tieu cua corpus khong chi la cat text nho ra, ma la tao retrieval unit co y nghia phau thuat, co provenance va co semantic tags.

Artifact dich den theo code hien tai:

- `docs/chunks/chunks_v3.jsonl`

Artifact local hien co:

- `docs/chunks/chunks.jsonl`

### 5.5. `scripts/retrieval.py`

Day la engine truy hoi tri thuc cho du an.

Lop trung tam:

- `SurgicalRetrieverV2`

Y tuong retrieval:

1. Tai chunks tu `CHUNKS_FILE`.
2. Index `contextualized_text` thay vi chi index `raw text`.
3. Uu tien `child chunks` cho retrieval.
4. Build `field-aware BM25` tren nhieu truong:
   - `contextualized_text`
   - `doc_title`
   - `section_title`
   - `chunk_type`
   - `anatomy_tags`
   - `risk_tags`
   - `phase_scope`
   - `instrument_tags`
   - `action_tags`
5. Neu co dense model thi build dense index bang SentenceTransformers + FAISS.
6. Neu bat reranker thi rerank top candidates.
7. Ap dung prior theo:
   - `question_type`
   - `collection`
   - `chunk_type`
   - `trust_tier`
   - `collection priority`
8. Expand child chunk len parent evidence de VLM doc evidence co context tot hon.

Script nay cung encode kien thuc domain bang:

- `QUESTION_TYPE_HINTS`
- `TERM_EXPANSIONS`
- `CLASS_TERM_MAP`
- `LOW_VALUE_SECTION_PATTERNS`
- `LOW_VALUE_TEXT_PATTERNS`

Y nghia:

- Retrieval cua du an khong phai la keyword search don gian.
- No da co logic domain-specific cho surgery va cho tung loai cau hoi.

### 5.6. `scripts/rag_vqa_pipeline.py`

Day la script chinh de chay `inference`.

Flow trong script nay:

1. Doc `questions_v3.json` tu `QUESTIONS_FILE`.
2. Voi tung cau hoi:
   - tim frame,
   - goi retriever,
   - tao prompt,
   - goi VLM,
   - parse output,
   - luu ket qua.
3. Cuoi cung ghi file `spike_results_v3.json`.

Pipeline nay ho tro 3 mode:

- `openai`
- `local_hf`
- `mock_vlm`

#### Doi voi retrieval + prompt

Ban sua gan nhat cua pipeline da duoc nang cap theo huong:

- prompt ngan hon,
- chi dua so evidence can thiet,
- co `question-type hint`,
- bo score am ra khoi prompt,
- co `compact retry` neu model tra output xau,
- co parser manh hon de bat:
  - bracketed confidence,
  - freeform defer,
  - `ANSWER: DEFER`,
  - template leak,
  - empty response,
  - garbage response.

#### Doi voi local VLM

Ban pipeline hien tai da support tot hon cho:

- `LLaVA`
- `Qwen2-VL`
- `Qwen2.5-VL`

Y nghia:

- Day la thanh phan hop nhat toan bo tri thuc retrieval, image reasoning, va output formatting.
- Chat luong thuc te cua he thong phu thuoc rat manh vao file nay.

### 5.7. `scripts/evaluate.py`

Day la script danh gia 3 tang cho output cua pipeline.

#### Tier 1 - Deterministic metrics

Bao gom:

- defer alignment
- format quality
- confidence distribution
- latency
- retrieval statistics
- breakdown theo `question_type` va `difficulty`

#### Tier 2 - Overlap metrics

Bao gom:

- BLEU-1
- BLEU-4
- METEOR
- ROUGE-L
- keyword recall

Muc tieu cua tang nay la do overlap text, nhung chi la mot proxy metric.

#### Tier 3 - VLM-as-Judge

Dung `Qwen2.5-VL-7B-Instruct` lam judge de cham:

- `correctness_score`
- `safety_score`
- `grounding_score`
- `defer_score`
- verdict:
  - `correct`
  - `acceptable`
  - `unsafe`
  - `should_defer`

Script nay con co kha nang:

- luu report markdown,
- luu metrics json,
- luu per-question csv,
- ve plots.

Y nghia:

- Day la script giup project vuot ra khoi muc `demo output` de tro thanh mot he thong co kha nang phan tich, benchmark va bao cao.

### 5.8. `scripts/download_hf_models.py`

Script nay dung de pre-download toan bo model Hugging Face can cho du an:

- dense retrieval model
- reranker model
- local VLM
- judge VLM

Nhu vay project co the:

- giam delay trong lan chay dau,
- tranh loi download giua chung,
- va tien cho chay offline neu model da duoc cache.

## 6. Cac file du lieu trung gian quan trong

De hieu du an, can xem toan bo he thong nhu mot chuoi bien doi artifact.

### 6.1. Artifact benchmark tu frame selection

- `frame_metadata.json`
  Mo ta quality, classes, difficulty, question-type score, defer score tren moi frame da chon.

- `question_blueprint.json`
  Khai bao frame nao gan voi `question_type`, `difficulty`, `should_defer` nao.

- `selection_summary.json`
  Tong hop phan bo benchmark.

- `validation_report.json`
  Bao cao sanity check sau khi selection xong.

### 6.2. Artifact annotation

- `questions.json` hoac `questions_v3.json`
  File dau vao cho pipeline inference.

- `retrieval_eval.json` hoac `retrieval_eval_v3.json`
  File phuc vu danh gia retrieval.

### 6.3. Artifact knowledge base

- `chunks.jsonl` hoac `chunks_v3.jsonl`
  Corpus retrieval da duoc cat chunk va gan metadata.

### 6.4. Artifact inference

- `spike_results_v3.json`
  Ket qua chay VQA tren 100 sample.

Moi row thuong co:

- `qid`
- `frame`
- `question`
- `retrieved_chunks`
- `retrieved_scores`
- `retrieved_previews`
- `raw_response`
- `parsed_answer`
- `confidence`
- `is_defer`
- `question_type`
- `difficulty`
- `should_defer`
- va trong ban pipeline moi co them:
  - `parse_flags`
  - `prompt_variant`
  - `prompt_evidence_count`
  - `local_model_kind`

## 7. Trang thai du an hien tai tren local

Day la phan rat quan trong de report trung thuc.

### 7.1. Nhung gi chung ta da build duoc

Chung ta da co:

1. Mot bo frame benchmark 100 sample da duoc chon.
2. Mot bo annotation 100 question co cau truc ro rang.
3. Mot corpus retrieval tu 7 tai lieu phau thuat, gom 1713 chunks.
4. Mot retrieval engine co:
   - BM25 field-aware,
   - dense retrieval,
   - reranking,
   - question-type priors,
   - child-to-parent evidence packaging.
5. Mot pipeline RAG-VQA co the:
   - doc frame,
   - retrieve evidence,
   - goi VLM,
   - parse output,
   - luu ket qua.
6. Mot he thong evaluation 3 tang.
7. Ban sua moi nhat cua pipeline da tang kha nang:
   - compact prompt,
   - parser robust,
   - retry compact mode,
   - support local Qwen-VL tot hon.

### 7.2. Trang thai artifact hien tai

Ve mat artifact, local hien dang o trang thai `co du rat nhieu thanh phan da build, nhung chua dong bo naming hoan toan giua code v3 va artifact cu`.

Cu the:

- Code hien tai mong cho:
  - `data/frames_v3`
  - `questions_v3.json`
  - `retrieval_eval_v3.json`
  - `chunks_v3.jsonl`

- Artifact local hien dang co:
  - `data/frames`
  - `questions.json`
  - `retrieval_eval.json`
  - `chunks.jsonl`

Dieu nay co nghia la:

- ve mat kien truc, pipeline da ro rang,
- ve mat artifact, local van dang o giai doan giao thoa giua `bo cu` va `code v3`.

### 7.3. Van de da duoc phat hien tu ket qua inference truoc do

Dot inference gan nhat bang local VLM cu cho thay cac van de lon:

1. rat nhieu output rong,
2. rat nhieu confidence parse thanh `unknown`,
3. he thong gan nhu khong detect duoc `defer`,
4. format output khong on dinh,
5. quality cua JSON ket qua rat xau de dua vao evaluation nghiem tuc.

Do la ly do `rag_vqa_pipeline.py` da duoc cai tien them ve:

- prompt design,
- parser logic,
- retry strategy,
- local model support.

## 8. Flow thuc te ma du an muon dat toi

Neu mo ta du an theo flow ly tuong, thi chuoi thuc thi se la:

1. Chay `frames_selection.py`
   de tao bo frame benchmark moi trong `data/frames_v3/`.

2. Chay `generate_annotations.py`
   de tao:
   - `questions_v3.json`
   - `retrieval_eval_v3.json`

3. Chay `build_corpus.py`
   de tao `chunks_v3.jsonl`.

4. Chay `rag_vqa_pipeline.py`
   de tao `spike_results_v3.json`.

5. Chay `evaluate.py`
   de tao:
   - report markdown
   - metrics json
   - per-question csv
   - plots

Nghia la:

- `frame benchmark` la nguon cua question set,
- `document corpus` la nguon cua retrieval,
- `pipeline` la noi gap nhau giua image va text knowledge,
- `evaluation` la noi ket qua duoc do luong.

## 9. Nhung diem manh cua he thong hien tai

### 9.1. Da co phan retrieval co tinh domain-specific

Du an khong dung retrieval tong quat thuong.
No co:

- question-type priors,
- collection priority,
- trust-tier boost,
- anatomy/risk/phase tag weighting,
- class-term expansion.

Day la mot diem manh rat quan trong de report.

### 9.2. Da co benchmark task formulation ro rang

Bo benchmark 100 sample da duoc thiet ke co chu dich:

- 5 loai cau hoi,
- 3 muc do kho,
- 25 sample should-defer.

Day la co so de danh gia he thong co y nghia hon rat nhieu so voi viec chon frame ngau nhien.

### 9.3. Da co evaluation 3 tang

He thong evaluation khong chi dung textual overlap, ma da co:

- metrics deterministic,
- overlap metrics,
- VLM-as-Judge co safety-aware rubric.

Day la mot diem rat tot cho bai report hoc thuat.

### 9.4. Da xac dinh ro bai toan `safe abstention`

He thong duoc thiet ke theo huong:

- khong ep model phai tra loi moi luc,
- ma cho phep model `defer` khi khong du bang chung.

Trong boi canh phau thuat, day la mot huong rat hop ly va co gia tri nghien cuu.

## 10. Nhung han che hien tai

De report trung thuc, day la nhung han che chinh:

1. Artifact local va naming convention `v3` chua dong bo hoan toan.
2. Dot inference cu bang `LLaVA-1.5-7B` cho output quality thap.
3. Khau answer generation hien van phu thuoc rat manh vao chat luong model VLM.
4. Gold answers trong annotation van mang tinh scaffold, chua phai final expert-validated references.
5. Mot so artifact `v3` chua duoc tai tao day du tren local moi.

## 11. Neu can giai thich ngan gon cho professor

Co the mo ta du an bang 4 cau sau:

1. Chung toi dang xay mot he thong `Surgical RAG-VQA` cho frame noi soi cat tui mat.
2. He thong ket hop `visual reasoning` voi `retrieved surgical knowledge` tu guideline/review de tra loi cau hoi ve anatomy, workflow, safety va risk.
3. Benchmark hien tai gom 100 frame duoc chon co chu dich, 5 nhom cau hoi, 3 muc do kho, va 25 truong hop can kha nang `defer`.
4. Chung toi da xay xong khung pipeline gom frame selection, annotation generation, corpus building, retrieval, VQA inference va evaluation 3 tang; hien tai dang o giai doan nang cap chat luong inference va dong bo artifact `v3`.

## 12. Ket luan

`SurgRAG-VQA` hien tai khong con la mot y tuong roi rac. No da la mot he thong co cau truc ro rang gom:

- benchmark construction,
- surgical document corpus,
- retrieval engine,
- VLM inference pipeline,
- va evaluation framework.

Noi dung quan trong nhat can nho:

1. Chung ta da build duoc gan nhu day du toan bo pipeline end-to-end.
2. Diem nghen lon nhat hien tai khong nam o retrieval architecture, ma nam o:
   - dong bo artifact `v3`,
   - va chat luong answer generation cua VLM.
3. Nen tang nghien cuu cua du an da co san:
   - benchmark,
   - corpus,
   - retrieval,
   - evaluation.

Vi vay, trang thai hien tai cua du an co the tom tat nhu sau:

`He thong da co day du khung ky thuat, da tao duoc artifact thuc va da chay duoc inference, nhung dang trong giai doan chuan hoa artifact va nang cap answer model/pipeline de co ket qua on dinh, sach va bao cao duoc theo chuan hoc thuat.`
