import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR         = PROJECT_ROOT / "data"
FRAMES_DIR       = DATA_DIR / "frames_v3"
ANNOTATIONS_DIR  = DATA_DIR / "annotations"

DOCS_DIR         = PROJECT_ROOT / "docs"
DOCS_RAW_DIR     = DOCS_DIR / "raw"
DOCS_CHUNKS_DIR  = DOCS_DIR / "chunks"

RESULTS_DIR      = PROJECT_ROOT / "results"

# Key files
QUESTIONS_FILE   = ANNOTATIONS_DIR / "questions_v3.json"
RETRIEVAL_EVAL_FILE = ANNOTATIONS_DIR / "retrieval_eval_v3.json"
CHUNKS_FILE = DOCS_CHUNKS_DIR / "chunks_v3.jsonl"
RESULTS_FILE     = RESULTS_DIR / "spike_results_v3.json"

# ─── Corpus Manifest ─────────────────────────────────────────────────
RAG_DOCUMENTS = [
    # ── Tier A: Core guidelines ───────────────────────────────────
    {
        "filename":       "sages_safe_chole.pdf",
        "doc_id":         "SAGES_GUIDE",
        "doc_title":      "SAGES Safe Cholecystectomy Multi-Society Practice Guideline",
        "source_type":    "guideline",
        "trust_tier":     "A",
        "collection":     "safe_chole_guideline",
        "priority":       1,
        "chunk_strategy": "section_aware",
        "chunk_size":     800,
        "tags_hint":      ["critical view of safety", "bile duct injury", "cystic duct",
                           "cystic artery", "hepatocystic triangle", "cholecystectomy",
                           "subtotal cholecystectomy", "bailout", "cholangiography"],
    },
    {
        "filename":       "tokyo_guidelines_2018_safe_steps.pdf",
        "doc_id":         "TOKYO_2018",
        "doc_title":      "Tokyo Guidelines 2018 — Safe Steps in Laparoscopic Cholecystectomy",
        "source_type":    "guideline",
        "trust_tier":     "A",
        "collection":     "safe_chole_guideline",
        "priority":       1,
        "chunk_strategy": "section_aware",
        "chunk_size":     800,
        "tags_hint":      ["acute cholecystitis", "difficult cholecystectomy", "bailout",
                           "fundus-first", "subtotal", "critical view of safety",
                           "inflammation", "fibrosis", "conversion"],
    },
    {
        "filename":       "wses_2020_bdi_guideline.pdf",
        "doc_id":         "WSES_BDI",
        "doc_title":      "WSES 2020 Guidelines — Bile Duct Injury During Cholecystectomy",
        "source_type":    "complication_guideline",
        "trust_tier":     "A",
        "collection":     "complication_management",
        "priority":       1,
        "chunk_strategy": "section_aware",
        "chunk_size":     800,
        "tags_hint":      ["bile duct injury", "recognition", "classification",
                           "repair", "referral", "bleeding", "bile leak"],
    },

    # ── Tier C: Anatomy & landmarks ───────────────────────────────
    {
        "filename":       "cvs_review.pdf",
        "doc_id":         "CVS_REVIEW",
        "doc_title":      "Critical View of Safety in Laparoscopic Cholecystectomy — Review",
        "source_type":    "anatomy_review",
        "trust_tier":     "C",
        "collection":     "biliary_anatomy_landmarks",
        "priority":       2,
        "chunk_strategy": "section_aware",
        "chunk_size":     600,
        "tags_hint":      ["critical view of safety", "criteria", "cystic plate",
                           "hepatocystic triangle", "two structures", "cystic duct",
                           "cystic artery", "misidentification", "common bile duct"],
    },
    {
        "filename":       "rouviere_sulcus.pdf",
        "doc_id":         "ROUVIERE",
        "doc_title":      "Rouviere's Sulcus — Anatomical Landmark for Safe Cholecystectomy",
        "source_type":    "anatomy_review",
        "trust_tier":     "C",
        "collection":     "biliary_anatomy_landmarks",
        "priority":       2,
        "chunk_strategy": "paragraph",
        "chunk_size":     600,
        "tags_hint":      ["rouviere sulcus", "landmark", "safe dissection zone",
                           "hepatocystic triangle", "right hepatic pedicle"],
    },

    # ── Tier D: Visual ontology ───────────────────────────────────
    {
        "filename":       "cholecseg8k_classes.pdf",
        "doc_id":         "CHOLECSEG",
        "doc_title":      "CholecSeg8k Segmentation Class Definitions",
        "source_type":    "ontology",
        "trust_tier":     "D",
        "collection":     "visual_ontology",
        "priority":       3,
        "chunk_strategy": "section_aware",
        "chunk_size":     500,
        "tags_hint":      ["segmentation", "gallbladder", "liver", "fat", "abdominal wall",
                           "gastrointestinal tract", "grasper", "hook", "hepatic vein",
                           "cystic duct", "blood"],
    },

    # ── Secondary: WHO ────────────────────────────────────────────
    {
        "filename":       "who_surgical_checklist.pdf",
        "doc_id":         "WHO_CHECK",
        "doc_title":      "WHO Surgical Safety Checklist",
        "source_type":    "safety_checklist",
        "trust_tier":     "B",
        "collection":     "general_or_safety",
        "priority":       4,
        "chunk_strategy": "section_aware",
        "chunk_size":     800,
        "tags_hint":      ["surgical safety", "checklist", "sign in", "time out",
                           "sign out", "team communication", "patient safety"],
    },
]

# ─── Chunking defaults ──────────────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 800
DEFAULT_CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH      = 80

# Token-based chunking (replaces char-based defaults for v3)
CHILD_CHUNK_TOKENS   = 250    # child: precise retrieval unit
PARENT_CHUNK_TOKENS  = 800    # parent: broader context for evidence packaging
CHILD_OVERLAP_TOKENS = 30
PARENT_OVERLAP_TOKENS = 80
MIN_CHUNK_TOKENS     = 30
 
# ─── Section detection patterns ──────────────────────────────────────
HEADING_PATTERNS = [
    r"^(?:Question|Step|Recommendation|Key Question)\s*\d+",
    r"^\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+[A-Z]",
    r"^[A-Z][A-Z\s\-]{3,}$",
    r"^(?:[A-Z][a-z]+(?:\s+(?:of|the|and|in|for|to|a|an|on|with|by|as|or|is|are|vs|at)\s+)?)+[A-Z][a-z]+",
]

JUNK_PATTERNS = [
    r"^\s*\d{1,4}\s*$",
    r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$",
    r"^\s*\d+\s*/\s*\d+\s*$",
    r"^\s*Downloaded from\b.*$",
    r"^\s*Copyright\s*©?\s*\d{4}.*$",
    r"^\s*All [Rr]ights [Rr]eserved.*$",
    r"^\s*This article is.*licensed under.*$",
    r"^\s*Open Access.*Creative Commons.*$",
    r"^\s*Content Provided by.*$",
    r"^\s*https?://\S+\s*$",
    r"^\s*doi:\s*\S+\s*$",
    r"^\s*DOI\s+\S+\s*$",
    r"^\s*\[\d+(?:[-–,]\s*\d+)*\]\s*$",
    r"^\s*$",
]

REFERENCE_MARKERS = [
    r"^\s*References?\s*$",
    r"^\s*REFERENCES?\s*$",
    r"^\s*Bibliography\s*$",
    r"^\s*Works Cited\s*$",
]

# ─── Retrieval ───────────────────────────────────────────────────────
DENSE_MODEL_NAME = os.environ.get(
    "DENSE_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
USE_RERANKER     = os.environ.get("USE_RERANKER", "0") == "1"
RERANKER_MODEL_NAME = os.environ.get(
    "RERANKER_MODEL_NAME",
    "BAAI/bge-reranker-large",
)
RERANK_TOP_N     = int(os.environ.get("RERANK_TOP_N", "20"))
HYBRID_ALPHA     = 0.6
RETRIEVAL_TOP_K  = 5
RETRIEVAL_MODE   = os.environ.get("RETRIEVAL_MODE", "hybrid")
HF_CACHE_DIR     = os.environ.get("HF_CACHE_DIR", "")
HF_LOCAL_FILES_ONLY = os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"

COLLECTION_PRIORITY = {
    "lc_step_by_step":           1.0,
    "biliary_anatomy_landmarks": 0.95,
    "safe_chole_guideline":      0.90,
    "complication_management":   0.85,
    "visual_ontology":           0.80,
    "general_or_safety":         0.60,
}

# ─── VLM ─────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN         = os.environ.get("HF_TOKEN", "")
VLM_PROVIDER     = os.environ.get(
    "VLM_PROVIDER",
    "openai" if OPENAI_API_KEY else "mock_vlm",
)
OPENAI_VLM_MODEL = os.environ.get("OPENAI_VLM_MODEL", "gpt-4o")
LOCAL_VLM_MODEL  = os.environ.get(
    "LOCAL_VLM_MODEL",
    "llava-hf/llava-1.5-7b-hf",
)
VLM_MODEL        = OPENAI_VLM_MODEL if VLM_PROVIDER == "openai" else LOCAL_VLM_MODEL
VLM_MAX_TOKENS   = 400
VLM_TEMPERATURE  = 0.2
LOCAL_VLM_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_VLM_MAX_NEW_TOKENS", "384"))

# ─── Evaluation Judge ────────────────────────────────────────────────
JUDGE_VLM_MODEL = os.environ.get(
    "JUDGE_VLM_MODEL",
    "Qwen/Qwen2.5-VL-7B-Instruct",
)

# ─── Evaluation ──────────────────────────────────────────────────────
CONFIDENCE_LEVELS = ["high", "medium", "low", "unknown"]

# ─── Ensure directories exist ────────────────────────────────────────
for d in [FRAMES_DIR, ANNOTATIONS_DIR, DOCS_RAW_DIR, DOCS_CHUNKS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
