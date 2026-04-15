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
CHUNKS_FILE = DOCS_CHUNKS_DIR / "chunks.jsonl"
RESULTS_FILE     = RESULTS_DIR / "spike_results_v3.json"

# ─── Corpus Manifest ─────────────────────────────────────────────────
# tier A = clinical guidelines + highly-cited clinical studies, 
# tier B = peer-reviewed reviews + cohort studies, 
# tier C = case reports + dataset papers + anatomy studies
RAG_DOCUMENTS = [
    {
        "filename":       "sages_safe_chole.pdf",
        "doc_id":         "SAGES_GUIDE",
        "doc_title":      "SAGES Safe Cholecystectomy Multi-Society Practice Guideline",
        "source_type":    "guideline",
        "doc_family":     "guideline",
        "trust_tier":     "A",
        "collection":     "safe_chole_guideline",
        "priority":       1,
        "chunk_strategy": "recommendation",
        "tags_hint":      ["critical view of safety", "bile duct injury", "subtotal cholecystectomy",
                           "bailout", "cholangiography", "hepatocystic triangle"],
    },
    {
        "filename":       "tokyo_guidelines_2018.pdf",
        "doc_id":         "TOKYO_2018",
        "doc_title":      "Tokyo Guidelines 2018 Safe Steps in Laparoscopic Cholecystectomy",
        "source_type":    "guideline",
        "doc_family":     "guideline",
        "trust_tier":     "A",
        "collection":     "safe_chole_guideline",
        "priority":       1,
        "chunk_strategy": "recommendation",
        "tags_hint":      ["acute cholecystitis", "difficult cholecystectomy", "safe steps",
                           "bailout", "fundus-first", "subtotal cholecystectomy"],
    },
    {
        "filename":       "wses_2020_bdi_guideline.pdf",
        "doc_id":         "WSES_BDI",
        "doc_title":      "WSES 2020 Guidelines for Bile Duct Injury During Cholecystectomy",
        "source_type":    "complication_guideline",
        "doc_family":     "guideline",
        "trust_tier":     "A",
        "collection":     "complication_management",
        "priority":       1,
        "chunk_strategy": "recommendation",
        "tags_hint":      ["bile duct injury", "strasberg classification", "bile leak",
                           "repair", "referral", "detection", "management"],
    },
    {
        "filename":       "The_difficult_laparoscopic_cholecystectomy.pdf",
        "doc_id":         "DIFFICULT_LC",
        "doc_title":      "The Difficult Laparoscopic Cholecystectomy",
        "source_type":    "clinical_review",
        "doc_family":     "review",
        "trust_tier":     "B",
        "collection":     "safe_chole_guideline",
        "priority":       1,
        "chunk_strategy": "clinical_review",
        "tags_hint":      ["difficult laparoscopic cholecystectomy", "acute cholecystitis",
                           "bailout", "conversion", "cholangiography", "safe dissection"],
    },
    {
        "filename":       "Laparoscopic_subtotal_cholecystectomy.pdf",
        "doc_id":         "LAP_SUBTOTAL",
        "doc_title":      "Laparoscopic Subtotal Cholecystectomy as a Bailout Procedure",
        "source_type":    "bailout_review",
        "doc_family":     "review",
        "trust_tier":     "B",
        "collection":     "safe_chole_guideline",
        "priority":       1,
        "chunk_strategy": "clinical_review",
        "tags_hint":      ["subtotal cholecystectomy", "bailout procedure", "difficult cholecystectomy",
                           "fenestrating", "reconstituting", "bile duct injury"],
    },
    {
        "filename":       "cvs_review.pdf",
        "doc_id":         "CVS_REVIEW",
        "doc_title":      "Critical View of Safety in Laparoscopic Cholecystectomy",
        "source_type":    "anatomy_review",
        "doc_family":     "review",
        "trust_tier":     "B",
        "collection":     "biliary_anatomy_landmarks",
        "priority":       2,
        "chunk_strategy": "clinical_review",
        "tags_hint":      ["critical view of safety", "cystic plate", "hepatocystic triangle",
                           "cystic duct", "cystic artery", "aberrant anatomy"],
    },
    {
        "filename":       "rouviere_sulcus.pdf",
        "doc_id":         "ROUVIERE",
        "doc_title":      "Rouviere Sulcus as an Anatomical Landmark for Safe Cholecystectomy",
        "source_type":    "anatomy_review",
        "doc_family":     "review",
        "trust_tier":     "B",
        "collection":     "biliary_anatomy_landmarks",
        "priority":       2,
        "chunk_strategy": "landmark_review",
        "tags_hint":      ["rouviere sulcus", "landmark", "safe dissection plane",
                           "hepatocystic triangle", "bile duct injury prevention"],
    },
    {
        "filename":       "Causes_and_Prevention_of_LBD_Injuries.pdf",
        "doc_id":         "CAUSES_LBDI",
        "doc_title":      "Causes and Prevention of Laparoscopic Bile Duct Injuries",
        "source_type":    "human_factors_review",
        "doc_family":     "human_factors",
        "trust_tier":     "B",
        "collection":     "complication_management",
        "priority":       2,
        "chunk_strategy": "human_factors",
        "tags_hint":      ["visual perceptual illusion", "misidentification", "bile duct injury",
                           "human factors", "cognitive psychology", "prevention rules"],
    },
    {
        "filename":       "Prevention_LBD_Injuries.pdf",
        "doc_id":         "PREVENT_LBDI",
        "doc_title":      "The Prevention of Laparoscopic Bile Duct Injuries",
        "source_type":    "human_factors_review",
        "doc_family":     "human_factors",
        "trust_tier":     "B",
        "collection":     "complication_management",
        "priority":       2,
        "chunk_strategy": "human_factors",
        "tags_hint":      ["bile duct injury", "visual perceptual illusion", "human factors",
                           "rules for prevention", "misperception", "poor visibility"],
    },
    {
        "filename":       "optical_illusion_the_cause_of_classical_bile_duct_injuries.pdf",
        "doc_id":         "OPTICAL_ILLUSION",
        "doc_title":      "Optical Illusion as a Cause of Classical Bile Duct Injuries",
        "source_type":    "human_factors_commentary",
        "doc_family":     "human_factors",
        "trust_tier":     "B",
        "collection":     "complication_management",
        "priority":       2,
        "chunk_strategy": "human_factors",
        "tags_hint":      ["optical illusion", "hartmann pouch", "misidentification",
                           "bile duct injury", "B-SAFE", "porta hepatis"],
    },
    {
        "filename":       "Formalizing_video_documentation.pdf",
        "doc_id":         "VIDEO_CVS",
        "doc_title":      "Formalizing Video Documentation of the Critical View of Safety",
        "source_type":    "video_documentation",
        "doc_family":     "dataset_paper",
        "trust_tier":     "C",
        "collection":     "visual_ontology",
        "priority":       3,
        "chunk_strategy": "dataset_paper",
        "tags_hint":      ["critical view of safety", "video reporting", "binary assessment",
                           "doublet view", "quality audit", "video clip"],
    },
    {
        "filename":       "Cholec80-CVS.pdf",
        "doc_id":         "CHOLEC80_CVS",
        "doc_title":      "Cholec80-CVS Open Dataset for Critical View of Safety Assessment",
        "source_type":    "dataset_paper",
        "doc_family":     "dataset_paper",
        "trust_tier":     "C",
        "collection":     "visual_ontology",
        "priority":       3,
        "chunk_strategy": "dataset_paper",
        "tags_hint":      ["critical view of safety", "annotation", "video dataset",
                           "2 points", "1 point", "0 points", "strasberg criteria"],
    },
    {
        "filename":       "The_Endoscapes_Dataset.pdf",
        "doc_id":         "ENDOSCAPES",
        "doc_title":      "The Endoscapes Dataset for Surgical Scene Understanding and CVS Assessment",
        "source_type":    "dataset_paper",
        "doc_family":     "dataset_paper",
        "trust_tier":     "C",
        "collection":     "visual_ontology",
        "priority":       3,
        "chunk_strategy": "dataset_paper",
        "tags_hint":      ["endoscapes", "segmentation", "object detection", "critical view of safety",
                           "bounding boxes", "surgical scene understanding"],
    },
    {
        "filename":       "The_SAGES_CVS_Challenge.pdf",
        "doc_id":         "SAGES_CVS_CHAL",
        "doc_title":      "The SAGES Critical View of Safety Challenge",
        "source_type":    "challenge_paper",
        "doc_family":     "dataset_paper",
        "trust_tier":     "C",
        "collection":     "visual_ontology",
        "priority":       3,
        "chunk_strategy": "dataset_paper",
        "tags_hint":      ["critical view of safety", "surgical quality assessment", "benchmark",
                           "challenge", "AI-assisted assessment", "workflow diversity"],
    },
]

# ─── Chunking defaults ──────────────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 800
DEFAULT_CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH      = 80

# Token-based chunking defaults for the active corpus build
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
    "Qwen/Qwen2.5-VL-7B-Instruct",
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
