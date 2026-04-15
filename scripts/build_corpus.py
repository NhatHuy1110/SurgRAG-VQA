"""
build_corpus.py — Corpus builder for SurgRAG-VQA.

Some features I've been built into this file:
  - Heading detection tightened — removed over-broad title-case pattern, added negative filters for author/affiliation/institution lines
  - Parent-child linking by sentence-span overlap, not equal distribution
  - Tag extraction uses word-boundary matching; dangerous short aliases (wash, port, clip, cut) removed or guarded with phrase context
  - Front matter cleaning: author blocks, affiliations, emails, URLs-in-text
  - Mojibake normalization (UTF-8 double-encoding artifacts)
  - Token-based chunking (tiktoken)
  - Parent-child hierarchy with parent_id / child_ids / sibling_ids
  - Page-level provenance (page_start, page_end, heading_path, section_id)
  - Synonym/alias-aware tag extraction
  - Contextualized chunk text
  - Section + document summaries
  - Built-in validation pass

Usage:
    python scripts/build_corpus.py
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CHUNKS_FILE,
    DOCS_RAW_DIR,
    RAG_DOCUMENTS,
    MIN_CHUNK_LENGTH,
    JUNK_PATTERNS,
    REFERENCE_MARKERS,
    DENSE_MODEL_NAME,
    HF_CACHE_DIR,
)

# ═══════════════════════════════════════════════════════════════════════
#  TOKENIZER
# ═══════════════════════════════════════════════════════════════════════

_tokenizer_max_len = 0

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    try:
        from transformers import AutoTokenizer
        _hf_tokenizer = AutoTokenizer.from_pretrained(
            DENSE_MODEL_NAME,
            cache_dir=HF_CACHE_DIR or None,
            local_files_only=True,
        )
        if 0 < getattr(_hf_tokenizer, "model_max_length", 0) < 100000:
            _tokenizer_max_len = int(_hf_tokenizer.model_max_length)
        print(f"[WARN] tiktoken not installed - using local tokenizer: {DENSE_MODEL_NAME}")

        def count_tokens(text: str) -> int:
            encoded = _hf_tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )
            return len(encoded["input_ids"])

    except Exception:
        print("[WARN] tokenizer libs unavailable - word-based fallback (~4 chars/token)")

        def count_tokens(text: str) -> int:
            return max(1, len(text) // 4)

CHILD_CHUNK_TOKENS   = 250
PARENT_CHUNK_TOKENS  = 800
CHILD_OVERLAP_TOKENS = 30
PARENT_OVERLAP_TOKENS = 80
MIN_CHUNK_TOKENS     = 30

if _tokenizer_max_len:
    PARENT_CHUNK_TOKENS = min(PARENT_CHUNK_TOKENS, max(320, _tokenizer_max_len - 32))
    CHILD_CHUNK_TOKENS = min(CHILD_CHUNK_TOKENS, max(160, _tokenizer_max_len // 2))


# ═══════════════════════════════════════════════════════════════════════
#  MOJIBAKE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════

_MOJIBAKE_MAP = {
    "\u00e2\x80\x99": "\u2019", "\u00e2\x80\x9c": "\u201c", "\u00e2\x80\x9d": "\u201d",
    "\u00e2\x80\x93": "\u2013", "\u00e2\x80\x94": "\u2014", "\u00e2\x80\x98": "\u2018",
    "\u00e2\x80\xa2": "\u2022", "\u00e2\x80\xa6": "\u2026",
    "\u00e2\x89\xa4": "\u2264", "\u00e2\x89\xa5": "\u2265",
    "\u00c3\u00a9": "\u00e9", "\u00c3\u00a8": "\u00e8", "\u00c3\u00b6": "\u00f6",
    "\u00c3\u00bc": "\u00fc", "\u00c3\u00a4": "\u00e4", "\u00c3\xad": "\u00ed",
    "\u00c3\u00b3": "\u00f3", "\u00c3\u00b1": "\u00f1",
    # latin-1 decoded variants
    "\u00e2\u20ac\u2122": "\u2019", "\u00e2\u20ac\u0153": "\u201c",
    "\u00e2\u20ac\x9d": "\u201d", "\u00e2\u20ac\u201c": "\u2013",
    "\u00e2\u20ac\u201d": "\u2014", "\u00e2\u20ac\u02dc": "\u2018",
    "\u00e2\u20ac\u00a2": "\u2022", "\u00e2\u20ac\u00a6": "\u2026",
    "\u00c2\xa0": " ",
}

_MOJIBAKE_SUSPECTS = (
    "\u00e2\x80", "\u00e2\u20ac", "\u00c3", "\u00c2",
)

def _mojibake_score(text: str) -> int:
    return sum(text.count(s) for s in _MOJIBAKE_SUSPECTS)

def _contains_mojibake(text: str) -> bool:
    return _mojibake_score(text or "") > 0

def normalize_mojibake(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")

    for _ in range(2):
        baseline = text
        for bad, good in _MOJIBAKE_MAP.items():
            text = text.replace(bad, good)

        candidates = []
        for enc in ("cp1252", "latin-1"):
            try:
                candidates.append(text.encode(enc).decode("utf-8"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue

        best = text
        best_score = _mojibake_score(text)
        for cand in candidates:
            cand = unicodedata.normalize("NFC", cand)
            for bad, good in _MOJIBAKE_MAP.items():
                cand = cand.replace(bad, good)
            cand_score = _mojibake_score(cand)
            if cand_score < best_score:
                best = cand
                best_score = cand_score

        text = best
        if text == baseline:
            break

    return text

def normalize_pdf_artifacts(text: str) -> str:
    text = (text or "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\u200b", "").replace("\u00ad", "")
    text = re.sub(r"/C\d{1,3}", " ", text)
    text = re.sub(r"([A-Za-z])\s*/C\d{1,3}([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"h\s*t\s*t\s*p\s*s?\s*:\s*/\s*/", "https://", text, flags=re.IGNORECASE)
    text = re.sub(r"\bVol\.\s*:\s*\(\d+\)\s*1\s*3\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"([A-Z])\s*\?\s*s", r"\1's", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

# ═══════════════════════════════════════════════════════════════════════
#  PDF EXTRACTION  (page-level)
# ═══════════════════════════════════════════════════════════════════════

def _extract_pypdf_pages(pdf_path: Path) -> list[dict]:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    return [{"page_num": i + 1, "text": p.extract_text() or ""}
            for i, p in enumerate(reader.pages) if (p.extract_text() or "").strip()]

def _extract_pdfplumber_pages(pdf_path: Path) -> list[dict]:
    import pdfplumber
    with pdfplumber.open(str(pdf_path)) as pdf:
        return [{"page_num": i + 1, "text": p.extract_text() or ""}
                for i, p in enumerate(pdf.pages) if (p.extract_text() or "").strip()]

def extract_pages(file_path: Path) -> list[dict]:
    if file_path.suffix.lower() in (".txt", ".md"):
        return [{"page_num": 1,
                 "text": file_path.read_text(encoding="utf-8", errors="replace")}]

    pages = []
    try:
        pages = _extract_pypdf_pages(file_path)
    except Exception as e:
        print(f"  [WARN] pypdf failed: {e}")

    total = " ".join(p["text"] for p in pages)
    if len(total.strip()) < 200:
        print(f"  [WARN] pypdf too short - trying pdfplumber...")
        try:
            pages = _extract_pdfplumber_pages(file_path)
        except Exception as e:
            print(f"  [WARN] pdfplumber also failed: {e}")

    if pages:
        total = " ".join(p["text"] for p in pages)
        ascii_ratio = sum(1 for c in total if ord(c) < 128) / max(len(total), 1)
        if ascii_ratio < 0.7:
            print(f"  [WARN] garbled ({ascii_ratio:.0%} ASCII) - trying pdfplumber...")
            try:
                pages = _extract_pdfplumber_pages(file_path)
            except Exception as e:
                print(f"  [WARN] pdfplumber also failed: {e}")

    for p in pages:
        p["text"] = normalize_mojibake(p["text"])
    return pages

# ═══════════════════════════════════════════════════════════════════════
#  TEXT CLEANING 
# ═══════════════════════════════════════════════════════════════════════

_junk_re = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in JUNK_PATTERNS]
_ref_re  = [re.compile(p, re.MULTILINE) for p in REFERENCE_MARKERS]

_FRONT_MATTER_PATTERNS = [
    re.compile(r"^\s*(?:authors?|institutions?|keywords?)\s*:?.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*\S+@\S+\.\S+\s*$", re.MULTILINE),
    re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+,\s*(?:MD|PhD|DO|FACS|FRCS|RN|MPH|MS|MBA|DrPH|MMedSc|MPhil)(?:,\s*(?:MD|PhD|DO|FACS|FRCS|RN|MPH|MS|MBA|DrPH|MMedSc|MPhil))*", re.MULTILINE),
    re.compile(r"^\s*(?:Department|Division|School|Faculty|Institute|Center|Centre)\s+of\s+.*$", re.MULTILINE),
    re.compile(r"^.*(?:University|Medical Center|Hospital|Institute|College|School of Medicine).*(?:,\s*[A-Z]{2}|,\s*[A-Z][a-z]+)[\s;]*$", re.MULTILINE),
    re.compile(r"^\s*(?:Correspond(?:ence|ing\s+author)|Address|Contact|E-?mail|Tel|Fax)\s*:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"\b(?:E-?mail|Email|Fax|Tel)\s*:\s*\S+", re.IGNORECASE),
    re.compile(r"^\s*(?:Received|Accepted|Published|Revised|Submitted)\s*:?\s*\d{1,2}\s+\w+\s+\d{4}.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"https?://\S+", re.MULTILINE),
    re.compile(r"\b(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/\S*)?", re.MULTILINE),
    re.compile(r"^\s*(?:DOI|doi)\s*:?\s*10\.\S+\s*$", re.MULTILINE),
    re.compile(r"^\s*\[\d+(?:\s*[-–,]\s*\d+)*\]\s*\.?\s*$", re.MULTILINE),
    re.compile(r"^\s*Presented at\b.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*(?:Disclosures?|Conflicts?\s+of\s+Interest|Funding)\s*:.*$", re.MULTILINE | re.IGNORECASE),
]

_ENDMATTER_NOISE_PATTERNS = [
    re.compile(r"^\s*(?:Authors?\s+and\s+contributors|Editors|Project team(?:\s+at)?|Additional acknowledgements|Contributors|Working group members|Additional consultants|Acknowledgements? for .*|Safe Surgery Saves Lives Programme Leader)\b.*$", re.MULTILINE | re.IGNORECASE),
]

_CONTENT_START_PATTERNS = [
    re.compile(r"^\s*(?:ABSTRACT|SUMMARY|BACKGROUND|INTRODUCTION|METHODS?|RESULTS?|DISCUSSION|CONCLUSION[S]?|APPENDIX)\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:Background|Introduction|Methods?|Results?|Discussion|Conclusion[s]?)\s*:", re.IGNORECASE),
    re.compile(r"^\s*(?:Question|Recommendation|Step|Key\s+Question)\s*\d+", re.IGNORECASE),
]
_INLINE_CONTENT_START_PATTERNS = [
    re.compile(r"\b(?:Abstract|Background|Introduction|Methods?|Results?|Discussion|Conclusion[s]?|Summary)\s*:", re.IGNORECASE),
    re.compile(r"\b(?:Question|Recommendation|Step|Key\s+Question)\s*\d+", re.IGNORECASE),
]

_REFERENCE_ENTRY_PATTERNS = [
    re.compile(r"^\s*\d{1,3}\.\s+.*\bet al\.\b.*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,3}\.\s+.*(?:19|20)\d{2}\s*;\s*\d", re.IGNORECASE),
    re.compile(r"^\s*\d{1,3}\.\s+.*(?:Journal|JAMA|Lancet|Commission|Organization|Guidelines?|protocol|anaesth|anesth)\b.*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,3}\.\s+[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)*(?:,\s*[A-Z]\.){1,3}.*$", re.IGNORECASE),
]

_TOC_LINE_PATTERNS = [
    re.compile(r"^\s*(?:Section\s+[IVX]+\.|Objective\s+\d+|Appendix\s+[A-Z]|Recommendations?|The problem:|The safe surgery saves lives|Improvement through|Organization of the guidelines|Before patient leaves operating room|Nurse verbally confirms|Surgeon, anaesthetist and nurse review).*\d{1,3}\s*$", re.IGNORECASE),
    re.compile(r"^\s*Printed in\s*$", re.IGNORECASE),
]
_HEADER_NOISE_PATTERNS = [
    re.compile(r"^\s*WHO Patient Safety\s*\|.*$", re.IGNORECASE),
    re.compile(r"^\s*Section\s+[IVXLC]+\.\s*\|.*$", re.IGNORECASE),
    re.compile(r"^\s*WHO Guidelines for Safe Surgery 2009\s*$", re.IGNORECASE),
]

def _remove_junk_lines(text: str) -> str:
    return "\n".join(l for l in text.split("\n")
                     if not any(p.match(l.strip()) for p in _junk_re)
                     and not any(p.match(l.strip()) for p in _HEADER_NOISE_PATTERNS))

def _remove_front_matter(text: str) -> str:
    for pat in _FRONT_MATTER_PATTERNS:
        text = pat.sub("", text)
    for pat in _ENDMATTER_NOISE_PATTERNS:
        text = pat.sub("", text)
    return text

def _looks_like_reference_entry(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(p.match(stripped) for p in _REFERENCE_ENTRY_PATTERNS):
        return True
    numbered = re.match(r"^\s*\d{1,3}\.\s+(.+)$", stripped)
    if not numbered:
        return False

    body = numbered.group(1).strip()
    if re.match(r"^(?:Appendix|Section|Step|Recommendation|Question|Key\s+Question|Objective|Aim|Goal)\b", body, re.IGNORECASE):
        return False

    comma_count = stripped.count(",")
    word_count = len(re.findall(r"[A-Za-z][A-Za-z'’\-]*", body))
    cues = [
        "et al.",
        "journal",
        "commission",
        "organization",
        "society",
        "association",
        "college",
        "task force",
        "guidelines",
        "protocol",
        "trial",
        "study",
        "points)",
    ]
    if comma_count >= 2:
        return True
    if word_count >= 6 and stripped.count(".") >= 2:
        first_clause, _, remainder = body.partition(". ")
        clause_words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", first_clause)
        titlecase_words = sum(1 for w in clause_words if re.match(r"^[A-Z][A-Za-z'’\-]+$", w))
        if remainder and (
            "," in first_clause
            or re.search(r"\b[A-Z]{1,4}\b", first_clause)
            or titlecase_words >= 3
        ):
            return True
    if any(cue in stripped.lower() for cue in cues):
        return True
    if re.search(r"(?:19|20)\d{2}", stripped):
        return True
    if re.search(r"\([^)]+(?:19|20)\d{2}[^)]*\)", stripped):
        return True
    if re.search(r"(?:,\s*[A-Z]\.){1,3}", stripped):
        return True
    return False

def _looks_like_toc_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(p.match(stripped) for p in _TOC_LINE_PATTERNS):
        return True
    if re.match(r"^.{10,120}\s+\d{1,3}\s*$", stripped):
        if re.search(r"(?:Section|Objective|Appendix|Recommendation|Introduction|Background|Methods|Results|Discussion|Conclusion)", stripped, re.IGNORECASE):
            return True
    return False

def _is_reference_heavy_text(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 5:
        return False
    ref_hits = sum(1 for l in lines[: min(12, len(lines))] if _looks_like_reference_entry(l))
    return ref_hits >= 4

def _is_reference_heavy_lines(lines: list[str]) -> bool:
    stripped = [line.strip() for line in lines if line.strip()]
    if len(stripped) < 6:
        return False
    sample = stripped[: min(16, len(stripped))]
    ref_hits = sum(1 for line in sample if _looks_like_reference_entry(line))
    numbered_hits = sum(1 for line in sample if re.match(r"^\d{1,3}\.\s+", line))
    return (
        ref_hits >= 6
        or (numbered_hits >= 8 and ref_hits >= max(5, numbered_hits - 2))
    )

def _is_toc_heavy_page(lines: list[str]) -> bool:
    stripped = [line.strip() for line in lines if line.strip()]
    if len(stripped) < 6:
        return False
    sample = stripped[: min(20, len(stripped))]
    toc_hits = sum(1 for line in sample if _looks_like_toc_line(line))
    page_tail_hits = sum(1 for line in sample if re.search(r"\s\d{1,3}\s*$", line))
    return toc_hits >= 5 or (toc_hits >= 3 and page_tail_hits >= 8)

def _is_front_matter_page(lines: list[str]) -> bool:
    stripped = [line.strip() for line in lines if line.strip()]
    if not stripped:
        return False
    sample = stripped[: min(25, len(stripped))]
    if len(sample) <= 2 and not any(
        re.match(r"^(?:Introduction|Background|Summary|Recommendations?|Appendix|Section\b|Objective\b|Step\b)", line, re.IGNORECASE)
        for line in sample
    ):
        return True
    signals = sum(
        1 for line in sample if re.search(
            r"\b(?:ISBN|Library Cataloguing|Requests for permission|WHO Press|World Health Organization|Published by|Copyright)\b",
            line,
            re.IGNORECASE,
        )
    )
    short_title_lines = sum(
        1 for line in sample[:5]
        if 1 <= len(re.findall(r"[A-Za-z][A-Za-z'’\-]*", line)) <= 8
    )
    return signals >= 2 or (signals >= 1 and short_title_lines >= 2 and len(sample) <= 30)

def _looks_like_front_matter_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _looks_like_reference_entry(stripped) or _looks_like_toc_line(stripped):
        return True
    if any(p.search(stripped) for p in _FRONT_MATTER_PATTERNS):
        return True
    if len(re.findall(r"\b(?:MD|PhD|DO|FACS|FRCS|RN|MPH|MS|MBA|DrPH|MMedSc|MPhil)\b", stripped, re.IGNORECASE)) >= 1:
        return True
    if stripped.count(";") >= 2 and re.search(r"(?:University|Hospital|Medical Center|Institute|School of Medicine)", stripped, re.IGNORECASE):
        return True
    if re.fullmatch(r"(?:\d+\s+){2,}\d+", stripped):
        return True
    return False

def _looks_like_body_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _looks_like_front_matter_line(stripped):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", stripped)
    return len(words) >= 8

def _trim_leading_front_matter(text: str) -> str:
    lines = text.split("\n")
    if not lines:
        return text

    window = lines[: min(80, len(lines))]
    front_hits = sum(1 for line in window if _looks_like_front_matter_line(line))
    has_anchor = any(any(p.match(line.strip()) for p in _CONTENT_START_PATTERNS) for line in window)
    if front_hits < 3 and not has_anchor:
        return text

    start_idx = 0
    saw_front = False
    for i, line in enumerate(window):
        stripped = line.strip()
        if any(p.match(stripped) for p in _CONTENT_START_PATTERNS):
            start_idx = i
            break
        if _looks_like_front_matter_line(stripped):
            saw_front = True
            start_idx = i + 1
            continue
        if saw_front and _looks_like_body_line(stripped):
            start_idx = i
            break

    trimmed = "\n".join(lines[start_idx:]).strip()
    return trimmed or text

def _trim_section_preamble(text: str) -> str:
    head = text[:4000]
    prefix = head[:2000]
    front_signals = sum(prefix.lower().count(tok) for tok in
                        ["authors", "institutions", "university", "hospital", "medical center", "email", "department"])
    if front_signals < 2:
        return text

    starts = []
    for pat in _INLINE_CONTENT_START_PATTERNS:
        m = pat.search(head)
        if m and m.start() > 40:
            starts.append(m.start())

    if not starts:
        lines = text.split("\n")
        toc_hits = sum(1 for line in lines[:60] if _looks_like_toc_line(line))
        if toc_hits < 5:
            return text

        for i, line in enumerate(lines[:80]):
            if _looks_like_body_line(line) and not _looks_like_toc_line(line):
                trimmed = "\n".join(lines[i:]).strip()
                return trimmed if len(trimmed) >= MIN_CHUNK_LENGTH else text
        return text

    trimmed = text[min(starts):].lstrip()
    return trimmed if len(trimmed) >= MIN_CHUNK_LENGTH else text

def _truncate_at_references(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if any(p.match(line.strip()) for p in _ref_re):
            trunc = "\n".join(lines[:i])
            if len(trunc.strip()) > 500:
                return trunc

    # Citation block fallback: if we hit a sustained run of reference-like entries,
    # cut the trailing bibliography even when there is no explicit "References" header.
    for i in range(len(lines)):
        window = [l for l in lines[i:i + 10] if l.strip()]
        if len(window) < 6:
            continue
        ref_hits = sum(1 for l in window if _looks_like_reference_entry(l))
        if ref_hits >= 6:
            trunc = "\n".join(lines[:i])
            if len(trunc.strip()) > 800:
                return trunc

    # Contributor/appendix tail fallback
    for i, line in enumerate(lines):
        stripped = line.strip()
        if any(p.match(stripped) for p in _ENDMATTER_NOISE_PATTERNS):
            trunc = "\n".join(lines[:i])
            if len(trunc.strip()) > 800:
                return trunc
    return text

def clean_text(text: str) -> str:
    text = text.replace("\x0c", "\n").replace("\x00", "")
    text = normalize_mojibake(text)
    text = normalize_pdf_artifacts(text)
    text = _remove_junk_lines(text)
    text = _remove_front_matter(text)
    text = _trim_leading_front_matter(text)
    text = _truncate_at_references(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# ═══════════════════════════════════════════════════════════════════════
#  HEADING DETECTION 
# ═══════════════════════════════════════════════════════════════════════

HEADING_PATTERNS = [
    r"^(?:Question|Step|Recommendation|Key\s+Question)\s*\d+",
    r"^(?:Section\s+[IVXLC]+|Objective\s+\d+|Appendix\s+[A-Z])\b",
    r"^(?:Introduction|Background|Methods?|Results?|Discussion|Conclusions?|Summary|Recommendations?)\b",
    r"^(?:How to use this manual|Modifying the Checklist|Introducing the Checklist into the operating room|Evaluating surgical care|The safe surgery saves lives approach|Organization of the guidelines|The World Health Organization Surgical Safety Checklist|Implementation Manual for the World Health Organization Surgical Safety Checklist)\b",
    r"^\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+[A-Z]",
    r"^[A-Z][A-Z\s\-]{4,}$",
]
_heading_re = [re.compile(p, re.MULTILINE) for p in HEADING_PATTERNS]

_HEADING_REJECT = [
    re.compile(r"(?:MD|PhD|DO|FACS|FRCS|MPH|RN|MBA|DrPH|MMedSc|MPhil)", re.IGNORECASE),
    re.compile(r"(?:University|Medical Center|Hospital|Institute|Department|College|School of)", re.IGNORECASE),
    re.compile(r"\S+@\S+\.\S+"),
    re.compile(r"(?:.*,){3,}"),
    re.compile(r",\s*[A-Z]{2}\s*[;,]"),
    re.compile(r"\([A-Z]{2,}\)"),
]

def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 150:
        return False
    if _looks_like_reference_entry(stripped) or _looks_like_toc_line(stripped):
        return False
    if re.match(r"^\d+(?:\.\d+){0,2}\.?\s+(?:In|On|At|The|A|An|This|These)\b", stripped):
        return False
    if not any(p.match(stripped) for p in _heading_re):
        return False
    if any(p.search(stripped) for p in _HEADING_REJECT):
        return False
    return True

# ═══════════════════════════════════════════════════════════════════════
#  SECTION DETECTION
# ═══════════════════════════════════════════════════════════════════════

def _update_heading_stack(stack: list[str], heading: str) -> list[str]:
    if re.match(r"^[A-Z][A-Z\s\-]{4,}$", heading):
        return []
    if re.match(r"^\d+\.\d+", heading):
        return stack[:1]
    if re.match(r"^\d+[\.\)]\s", heading):
        return []
    return stack[:2]

_SECTION_TITLE_DROP = [
    re.compile(r"^\s*(?:THIEME|GUIDELINE|FEATURE|NA NA|BMI BMI|SECONDS FROM)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s+[A-Z][A-Za-z]+(?:et al\.?|[A-Za-z])+\d{4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s+Points?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\.\s*[A-Z][^a-z]{0,3}\s*$"),
    re.compile(r"^\s*\d+\s*J\b.*\d{4}.*$", re.IGNORECASE),
]

def _canonicalize_heading(heading: str) -> str:
    heading = normalize_pdf_artifacts(normalize_mojibake((heading or "").strip()))
    if not heading:
        return ""
    if re.fullmatch(r"(?:[A-Z]\s+){4,}[A-Z]", heading):
        compact = heading.replace(" ", "")
        if compact.upper() == "ABSTRACT":
            return "ABSTRACT"
        return compact.title()
    heading = re.sub(r"\s{2,}", " ", heading)
    heading = re.sub(r"^[\W_]+|[\W_]+$", "", heading)
    return heading[:160].strip()

def _append_section(
    sections: list[dict],
    heading_stack: list[str],
    title: str,
    lines: list[str],
    page_start: int,
    page_end: int,
    sec_counter: int,
    content_chars: int,
) -> tuple[int, int]:
    title = _canonicalize_heading(title)
    body = _trim_section_preamble("\n".join(lines).strip())
    if len(body) <= MIN_CHUNK_LENGTH:
        return sec_counter, content_chars
    if not title or any(p.match(title) for p in _SECTION_TITLE_DROP):
        if len(body) < 240:
            return sec_counter, content_chars
        title = "Section"
    if _looks_like_reference_entry(title) or _is_reference_heavy_text(body):
        return sec_counter, content_chars

    sec_counter += 1
    content_chars += len(body)
    sections.append({
        "title": title,
        "text": body,
        "page_start": page_start,
        "page_end": page_end,
        "heading_path": " > ".join(heading_stack + [title]),
        "section_id": f"SEC_{sec_counter:03d}",
    })
    return sec_counter, content_chars

def detect_sections_with_pages(pages: list[dict]) -> list[dict]:
    sections, heading_stack = [], []
    cur_title, cur_lines = "Introduction", []
    cur_page_start = pages[0]["page_num"] if pages else 1
    cur_page_end = cur_page_start
    sec_counter = 0
    content_chars = 0
    ref_run = 0

    for pg in pages:
        pn = pg["page_num"]
        page_lines = [line for line in clean_text(pg["text"]).split("\n") if line.strip()]
        if not page_lines:
            continue

        if not sections and not cur_lines and _is_front_matter_page(page_lines):
            continue

        if _is_toc_heavy_page(page_lines):
            continue

        buffered_chars = content_chars + sum(len(line) for line in cur_lines)
        if buffered_chars > 1200 and _is_reference_heavy_lines(page_lines):
            if cur_lines:
                sec_counter, content_chars = _append_section(
                    sections, heading_stack, cur_title, cur_lines,
                    cur_page_start, cur_page_end, sec_counter, content_chars,
                )
                cur_lines = []
            ref_run = 0
            continue

        for idx, line in enumerate(page_lines):
            stripped = line.strip()

            if _looks_like_reference_entry(stripped):
                ref_run += 1
            else:
                ref_run = 0

            remaining_lines = page_lines[idx: idx + 12]
            buffered_chars = content_chars + sum(len(chunk_line) for chunk_line in cur_lines)
            if buffered_chars > 1200 and (
                ref_run >= 3 or _is_reference_heavy_lines(remaining_lines)
            ):
                if cur_lines:
                    sec_counter, content_chars = _append_section(
                        sections, heading_stack, cur_title, cur_lines,
                        cur_page_start, cur_page_end, sec_counter, content_chars,
                    )
                    cur_lines = []
                ref_run = 0
                break

            if any(p.match(stripped) for p in _ENDMATTER_NOISE_PATTERNS) and content_chars > 1200:
                if cur_lines:
                    sec_counter, content_chars = _append_section(
                        sections, heading_stack, cur_title, cur_lines,
                        cur_page_start, cur_page_end, sec_counter, content_chars,
                    )
                    cur_lines = []
                break

            if _is_heading(line) and cur_lines:
                sec_counter, content_chars = _append_section(
                    sections, heading_stack, cur_title, cur_lines,
                    cur_page_start, cur_page_end, sec_counter, content_chars,
                )
                heading_stack = _update_heading_stack(heading_stack, line.strip())
                cur_title, cur_lines, cur_page_start, cur_page_end = line.strip(), [], pn, pn
            else:
                cur_lines.append(line)
                cur_page_end = pn

    if cur_lines:
        sec_counter, content_chars = _append_section(
            sections, heading_stack, cur_title, cur_lines,
            cur_page_start, cur_page_end, sec_counter, content_chars,
        )

    if not sections:
        full = _trim_section_preamble(clean_text("\n".join(p["text"] for p in pages)))
        sections = [{"title": "Full Document", "text": full,
                     "page_start": pages[0]["page_num"] if pages else 1,
                     "page_end": pages[-1]["page_num"] if pages else 1,
                     "heading_path": "Full Document",
                     "section_id": "SEC_001"}]
    sections = [
        s for s in sections
        if not _looks_like_reference_entry(s["title"])
        and not _is_reference_heavy_text(s["text"])
    ]
    return sections

DATASET_KEEP_TITLE_PATTERNS = [
    re.compile(r"\babstract\b", re.IGNORECASE),
    re.compile(r"\bintroduction\b", re.IGNORECASE),
    re.compile(r"\bbackground\b", re.IGNORECASE),
    re.compile(r"\bcontext\b", re.IGNORECASE),
    re.compile(r"\bdataset\b", re.IGNORECASE),
    re.compile(r"\bannotation\b", re.IGNORECASE),
    re.compile(r"\bbenchmark\b", re.IGNORECASE),
    re.compile(r"\bcvs\b", re.IGNORECASE),
    re.compile(r"\bcritical view of safety\b", re.IGNORECASE),
    re.compile(r"\bdiscussion\b", re.IGNORECASE),
    re.compile(r"\bconclusion\b", re.IGNORECASE),
]

DROP_TITLE_PATTERNS = [
    re.compile(r"\brelated work\b", re.IGNORECASE),
    re.compile(r"\backnowledg", re.IGNORECASE),
    re.compile(r"\bappendix\b", re.IGNORECASE),
    re.compile(r"\bethic", re.IGNORECASE),
    re.compile(r"\bconflict", re.IGNORECASE),
    re.compile(r"\bfunding\b", re.IGNORECASE),
    re.compile(r"\bavailability of data\b", re.IGNORECASE),
    re.compile(r"\bauthor", re.IGNORECASE),
    re.compile(r"\bbiomedical challenges\b", re.IGNORECASE),
]

LOW_VALUE_TEXT_PATTERNS = [
    re.compile(r"\ball rights reserved\b", re.IGNORECASE),
    re.compile(r"\bcreativecommons\b", re.IGNORECASE),
    re.compile(r"\bcorrespondence to\b", re.IGNORECASE),
    re.compile(r"\bpublished online\b", re.IGNORECASE),
    re.compile(r"\bdoi:\b", re.IGNORECASE),
]

def _looks_like_table_block(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    numericish = sum(
        1 for line in lines[:12]
        if re.search(r"\b\d+(?:\.\d+)?\b", line) and len(re.findall(r"[A-Za-z]+", line)) <= 6
    )
    return numericish >= max(4, len(lines[:12]) // 2)


def _looks_like_study_table(text: str) -> bool:
    sample = text[:4000]
    na_hits = len(re.findall(r"\bNA\b", sample))
    year_hits = len(re.findall(r"\b20\d{2}\b", sample))
    bracket_hits = len(re.findall(r"\(\d+\)", sample))
    table_hits = len(re.findall(r"\bTable\s+\d+\b", sample, re.IGNORECASE))
    return (na_hits >= 4 and year_hits >= 3 and bracket_hits >= 2) or (table_hits >= 1 and na_hits >= 3 and year_hits >= 2)


def _looks_like_author_block(text: str) -> bool:
    sample = text[:1600]
    aff_hits = len(re.findall(r"\b(?:university|hospital|institute|medical school|department of|france|germany|usa|italy|china|canada)\b", sample, re.IGNORECASE))
    comma_hits = sample.count(",")
    lineish_hits = sample.count("\n")
    return aff_hits >= 4 and comma_hits >= 12 and lineish_hits >= 6

def _sanitize_section_text(text: str) -> str:
    text = normalize_pdf_artifacts(text)
    text = re.sub(r"\b(?:Open Access|REVIEW)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:Received|Accepted|Published online)\s*:[^\n]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAddress for correspondence\b.*?(?=(?:Abstract|Introduction|Background|Objective|Methods?|Results?|Discussion|Conclusions?|We sought))", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\bCorrespondence to\b.*?(?=(?:Abstract|Introduction|Background|Objective|Methods?|Results?|Discussion|Conclusions?|We sought))", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\bSee the Terms and Conditions\b.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    head = text[:5000]
    metadata_prefix = re.search(
        r"(?:address for correspondence|correspondence to|department of|hospital|university|received|accepted|published online|open access)",
        head[:2000],
        re.IGNORECASE,
    )
    if metadata_prefix:
        for tok in ["abstract", "introduction", "background", "method", "result", "discussion", "conclusion", "objective", "we sought"]:
            pos = head.lower().find(tok)
            if 80 <= pos <= 4500:
                candidate = text[pos:].strip()
                if len(candidate) >= 220 and candidate[:1].isupper():
                    text = candidate
                    break
    text = re.sub(r"\b(?:Author contribution|Declaration of competing interest|Acknowledg?ments?|Funding|Provenance and peer review|Consent|Sources of funding for your research|Ethical approval|Registration of research studies)\b.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\bReferences\b.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _trim_dataset_frontmatter(text: str) -> str:
    head = text[:6000]
    for tok in [r"\bA\s*B\s*S\s*T\s*R\s*A\s*C\s*T\b", r"\bAbstract\b", r"\b1\.\s*Introduction\b", r"\bIntroduction\b"]:
        m = re.search(tok, head, re.IGNORECASE)
        if m and m.start() >= 180:
            candidate = text[m.start():].strip()
            if len(candidate) >= 220:
                return candidate
    return text

def _section_signal_score(section: dict, doc_config: dict) -> int:
    title = (section.get("title") or "").lower()
    text = (section.get("text") or "").lower()
    score = 0
    for hint in doc_config.get("tags_hint", []):
        hint = hint.lower()
        if hint in title:
            score += 2
        elif hint in text:
            score += 1
    return score


def _infer_section_title(text: str, fallback: str) -> str:
    text = normalize_pdf_artifacts(text or "")
    start = text[:800].strip()
    if not start:
        return fallback

    m = re.match(r"^\s*\d+(?:\.\d+)*\.\s*([A-Z][A-Za-z][^\n]{3,120})", start)
    if m:
        return m.group(1).strip(" -:;,.")

    for label in [
        "Abstract", "Background", "Introduction", "Methods", "Results",
        "Discussion", "Conclusion", "Conclusions", "Summary",
    ]:
        if re.match(rf"^\b{label}\b[:\s-]", start, re.IGNORECASE):
            return "Conclusions" if label == "Conclusion" else label

    first_line = start.split("\n", 1)[0].strip()
    first_line = re.sub(r"\[[^\]]+\]", " ", first_line)
    first_line = re.sub(r"\s+", " ", first_line).strip(" -:;,.")
    if len(first_line) > 20:
        words = first_line.split()
        return " ".join(words[:12]).strip(" -:;,.")

    first_sentence = re.split(r"(?<=[.!?])\s+", start, maxsplit=1)[0]
    first_sentence = re.sub(r"\[[^\]]+\]", " ", first_sentence)
    first_sentence = re.sub(r"\s+", " ", first_sentence).strip(" -:;,.")
    words = first_sentence.split()
    if len(words) >= 4:
        return " ".join(words[:12]).strip(" -:;,.")
    return fallback

def _should_keep_section(section: dict, doc_config: dict) -> bool:
    title = (section.get("title") or "").strip()
    text = (section.get("text") or "").strip()
    family = doc_config.get("doc_family", "")
    source_type = doc_config.get("source_type", "")

    if len(text) < MIN_CHUNK_LENGTH:
        return False
    if any(p.search(title) for p in DROP_TITLE_PATTERNS):
        return False
    if any(p.search(text) for p in LOW_VALUE_TEXT_PATTERNS) and len(text) < 400:
        return False
    signal_score = _section_signal_score(section, doc_config)
    if _looks_like_table_block(text) and signal_score < 3:
        return False
    if _looks_like_study_table(text):
        return False

    if family == "dataset_paper":
        if _looks_like_author_block(text):
            return False
        keep = any(p.search(title) for p in DATASET_KEEP_TITLE_PATTERNS)
        if keep:
            return True
        if "critical view of safety" in text.lower() or "annotation" in text.lower():
            return True
        return signal_score >= 2 and len(text) >= 220

    if family == "human_factors":
        if re.search(r"\b(?:introduction|methods?|results?|discussion|conclusions?|summary)\b", title, re.IGNORECASE):
            return True
        if re.search(r"\b(?:illusion|misidentification|misperception|human factors?|error|prevention|rule)\b", text, re.IGNORECASE):
            return True
        return signal_score >= 2

    if source_type in {"guideline", "complication_guideline"}:
        if re.search(r"\b(?:recommendation|summary of evidence|background|question|step|conclusion)\b", title, re.IGNORECASE):
            return True
        if re.search(r"\b(?:recommend|we suggest|we recommend|safe steps?|bailout|critical view of safety)\b", text, re.IGNORECASE):
            return True
        return signal_score >= 2

    if source_type in {"anatomy_review", "clinical_review", "bailout_review"}:
        if re.search(r"\b(?:introduction|discussion|conclusion|results|methods?|case presentation)\b", title, re.IGNORECASE):
            return True
        return signal_score >= 1

    return True

def filter_sections_for_document(sections: list[dict], doc_config: dict) -> list[dict]:
    if doc_config.get("doc_id") == "ROUVIERE":
        for section in sections:
            full_text = _sanitize_section_text(section.get("text", ""))
            m = re.search(r"\b(?:Abstract|Introduction)\b", full_text, re.IGNORECASE)
            if m:
                curated = full_text[m.start():]
                curated = re.sub(r"\b2\.1\.\s*Registration\b.*", "", curated, flags=re.IGNORECASE | re.DOTALL)
                curated = curated.strip()
                if len(curated) >= 300:
                    sections = [{
                        **section,
                        "title": "Rouviere Sulcus Landmark",
                        "text": curated,
                        "heading_path": "Rouviere Sulcus Landmark",
                    }]
                    break

    filtered = []
    sec_counter = 0
    for section in sections:
        text = _sanitize_section_text(section.get("text", ""))
        if doc_config.get("doc_family") == "dataset_paper":
            text = _trim_dataset_frontmatter(text)
        title = _canonicalize_heading(section.get("title", ""))
        if not title or title == "Section" or (
            title == "Introduction"
            and not re.match(r"^(?:Introduction|Background|Abstract)\b", text, re.IGNORECASE)
        ):
            title = _infer_section_title(text, title or "Section")
        normalized = {
            **section,
            "title": title or "Section",
            "text": text,
            "heading_path": _canonicalize_heading(section.get("heading_path", "")) or title or "Section",
        }
        if not _should_keep_section(normalized, doc_config):
            continue
        sec_counter += 1
        normalized["section_id"] = f"SEC_{sec_counter:03d}"
        filtered.append(normalized)
    return filtered

# ═══════════════════════════════════════════════════════════════════════
#  SENTENCE SPLITTING
# ═══════════════════════════════════════════════════════════════════════

_ABBREVS = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "vs.", "etc.",
            "i.e.", "e.g.", "Fig.", "fig.", "Vol.", "vol.",
            "No.", "no.", "Ref.", "ref.", "approx.", "ca.",
            "et al.", "Sr.", "Jr."]

def sentence_split(text: str) -> list[str]:
    sentences = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        protected = para
        for abbr in _ABBREVS:
            protected = protected.replace(abbr, abbr.replace(".", "§"))
        for part in re.split(r'(?<=[.!?])\s+', protected):
            restored = part.replace("§", ".").strip()
            if restored:
                sentences.append(restored)
    return sentences

# ═══════════════════════════════════════════════════════════════════════
#  TAG EXTRACTION 
# ═══════════════════════════════════════════════════════════════════════

def _wb(phrase: str) -> re.Pattern:
    """Compile phrase with word boundaries."""
    return re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)

# ── Anatomy ──
ANATOMY_KEYWORDS = {
    "gallbladder", "cystic duct", "common bile duct", "common hepatic duct",
    "hepatic duct", "cystic artery", "hepatic artery", "right hepatic artery",
    "portal vein", "liver bed", "gallbladder fossa", "fundus", "infundibulum",
    "hartmann pouch", "gallbladder neck", "serosa", "peritoneum",
    "hepatocystic triangle", "cystic plate", "critical view of safety",
    "rouviere sulcus", "segment iv", "segment v",
    "omentum", "duodenum", "colon", "bile duct", "biliary",
    "ampulla", "sphincter of oddi",
}
_ANAT_PAT = {kw: _wb(kw) for kw in ANATOMY_KEYWORDS}
_ANAT_ALIAS = {
    "cvs": ("critical view of safety", _wb("cvs")),
    "calot's triangle": ("hepatocystic triangle", _wb("calot's triangle")),
    "calot triangle":   ("hepatocystic triangle", _wb("calot triangle")),
    "triangle of calot":("hepatocystic triangle", _wb("triangle of calot")),
    "hartmann's pouch": ("hartmann pouch",        _wb("hartmann's pouch")),
    "rouviere's sulcus":("rouviere sulcus",       _wb("rouviere's sulcus")),
    "gallbladder bed":  ("liver bed",             _wb("gallbladder bed")),
}

# ── Instruments ── (port, clip, hook, camera REMOVED as standalone)
INSTRUMENT_KEYWORDS = {
    "grasper", "dissector", "maryland dissector", "hook electrocautery",
    "electrocautery", "monopolar", "bipolar", "clip applier", "scissors",
    "irrigator", "suction", "trocar", "laparoscope",
    "endoscope", "retractor", "specimen bag",
    "cholangiography catheter", "drain",
}
_INST_PAT = {kw: _wb(kw) for kw in INSTRUMENT_KEYWORDS}
_INST_ALIAS = {
    "maryland dissector": ("maryland dissector", _wb("maryland dissector")),
    "l-hook":             ("hook electrocautery",_wb("l-hook")),
    "l hook":             ("hook electrocautery",_wb(r"l hook")),
    "hook cautery":       ("hook electrocautery",_wb("hook cautery")),
    "diathermy":          ("electrocautery",     _wb("diathermy")),
    "endo bag":           ("specimen bag",       _wb("endo bag")),
    "endobag":            ("specimen bag",       _wb("endobag")),
    "trocar port":        ("trocar",             _wb("trocar port")),
    "port placement":     ("trocar",             _wb("port placement")),
    "port site":          ("trocar",             _wb("port site")),
}

# ── Actions ── (clip, cut, wash REMOVED as standalone)
ACTION_KEYWORDS = {
    "dissect", "dissection", "retract", "retraction", "clipping",
    "divide", "division", "cauterize", "coagulate", "irrigate",
    "grasping", "expose", "exposure", "peel", "strip",
    "incise", "incision", "extraction",
}
_ACT_PAT = {kw: _wb(kw) for kw in ACTION_KEYWORDS}
_ACT_ALIAS = {
    "ligation":    ("clipping",   _wb("ligation")),
    "ligate":      ("clipping",   _wb("ligate")),
    "coagulation": ("coagulate",  _wb("coagulation")),
}

# ── Risks ──
RISK_KEYWORDS = {
    "bile duct injury", "bleeding", "hemorrhage", "hemostasis",
    "bile leak", "bile spillage", "perforation", "thermal injury",
    "misidentification", "wrong structure", "conversion",
    "inflammation", "fibrosis", "adhesion", "cholecystitis",
    "empyema", "gangrene", "necrosis",
}
_RISK_PAT = {kw: _wb(kw) for kw in RISK_KEYWORDS}
_RISK_ALIAS = {
    "bile duct injuries":       ("bile duct injury", _wb("bile duct injuries")),
    "vascular injury":          ("bleeding",         _wb("vascular injury")),
    "haemorrhage":              ("hemorrhage",       _wb("haemorrhage")),
    "haemostasis":              ("hemostasis",       _wb("haemostasis")),
    "gallstone spillage":       ("bile spillage",    _wb("gallstone spillage")),
    "burn injury":              ("thermal injury",   _wb("burn injury")),
    "wrong duct":               ("misidentification",_wb("wrong duct")),
    "acute cholecystitis":      ("cholecystitis",    _wb("acute cholecystitis")),
    "gangrenous cholecystitis": ("gangrene",         _wb("gangrenous cholecystitis")),
}

# ── Phases ──
PHASE_KEYWORDS = {
    "setup":                  ["port placement", "trocar", "insufflation",
                               "pneumoperitoneum"],
    "exposure":               ["retraction", "visualization", "camera position"],
    "dissection":             ["hepatocystic", "peritoneal", "dissect",
                               "cvs",
                               "critical view of safety"],
    "clipping_division":      ["clip applier", "divide", "cystic duct",
                               "cystic artery", "ligate", "transect"],
    "gallbladder_separation": ["liver bed", "gallbladder fossa", "peel",
                               "electrocautery", "separation"],
    "extraction":             ["specimen bag", "extraction", "retrieve"],
    "hemostasis_inspection":  ["hemostasis", "bile leak"],
}
_PHASE_PAT = {ph: [_wb(kw) for kw in kws]
              for ph, kws in PHASE_KEYWORDS.items()}

def _match_kw_alias(text: str, kw_pats: dict, alias_pats: dict) -> list[str]:
    found = set()
    for kw, pat in kw_pats.items():
        if pat.search(text):
            found.add(kw)
    for alias, (canon, pat) in alias_pats.items():
        if pat.search(text):
            found.add(canon)
    return sorted(found)

def extract_tags(text: str) -> dict:
    return {
        "anatomy_tags":    _match_kw_alias(text, _ANAT_PAT, _ANAT_ALIAS),
        "instrument_tags": _match_kw_alias(text, _INST_PAT, _INST_ALIAS),
        "action_tags":     _match_kw_alias(text, _ACT_PAT, _ACT_ALIAS),
        "risk_tags":       _match_kw_alias(text, _RISK_PAT, _RISK_ALIAS),
        "phase_scope":     [ph for ph, pats in _PHASE_PAT.items()
                            if any(p.search(text) for p in pats)],
    }

# ═══════════════════════════════════════════════════════════════════════
#  TOKEN-AWARE PACKING
# ═══════════════════════════════════════════════════════════════════════

def _split_overlong_text(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    pieces, current = [], []
    for word in words:
        candidate = " ".join(current + [word])
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        pieces.append(" ".join(current))
    return pieces

def _pack_sentences_tokens(sentences: list[str], max_tokens: int,
                           overlap_tokens: int) -> list[str]:
    chunks, current, cur_tok = [], [], 0
    expanded = []
    for sent in sentences:
        if count_tokens(sent) > max_tokens:
            expanded.extend(_split_overlong_text(sent, max_tokens))
        else:
            expanded.append(sent)

    for sent in expanded:
        st = count_tokens(sent)
        if cur_tok + st > max_tokens and current:
            chunks.append(" ".join(current))
            ov_sents, ov_tok = [], 0
            for s in reversed(current):
                t = count_tokens(s)
                if ov_tok + t > overlap_tokens:
                    break
                ov_sents.insert(0, s)
                ov_tok += t
            current, cur_tok = ov_sents, ov_tok
            if cur_tok + st > max_tokens:
                current, cur_tok = [], 0
        current.append(sent)
        cur_tok += st
    if current:
        chunks.append(" ".join(current))
    return chunks

# ═══════════════════════════════════════════════════════════════════════
#  CHUNKING 
# ═══════════════════════════════════════════════════════════════════════

def _section_meta(section: dict) -> dict:
    return {k: section.get(k) for k in
            ("page_start", "page_end", "heading_path", "section_id")}

def _build_parent_child_for_section(section: dict) -> list[dict]:
    """Build parent + child chunks with sentence-overlap-based linking."""
    sents = sentence_split(section["text"])
    if not sents:
        return []

    meta = _section_meta(section)
    title = section["title"]

    parent_texts = _pack_sentences_tokens(sents, PARENT_CHUNK_TOKENS,
                                          PARENT_OVERLAP_TOKENS)
    child_texts  = _pack_sentences_tokens(sents, CHILD_CHUNK_TOKENS,
                                          CHILD_OVERLAP_TOKENS)

    parents = []
    for pi, pt in enumerate(parent_texts):
        if count_tokens(pt) < MIN_CHUNK_TOKENS:
            continue
        parents.append({**meta, "section_title": title,
                        "text": pt, "level": "parent",
                        "_local_idx": pi,
                        "_sents": set(sentence_split(pt))})

    if not parents and child_texts:
        fallback_parent = max(child_texts, key=count_tokens)
        parents.append({**meta, "section_title": title,
                        "text": fallback_parent, "level": "parent",
                        "_local_idx": 0,
                        "_sents": set(sentence_split(fallback_parent))})

    children = []
    for ci, ct in enumerate(child_texts):
        if count_tokens(ct) < MIN_CHUNK_TOKENS:
            continue
        child = {**meta, "section_title": title,
                 "text": ct, "level": "child", "_local_idx": ci}
        # Find best-matching parent by sentence overlap
        child_sents = set(sentence_split(ct))
        best_idx, best_ov = None, 0
        for p in parents:
            ov = len(child_sents & p["_sents"])
            if ov > best_ov:
                best_ov = ov
                best_idx = p["_local_idx"]
        if best_idx is None:
            fallback_idx = max((p["_local_idx"] for p in parents), default=-1) + 1
            parents.append({**meta, "section_title": title,
                            "text": ct, "level": "parent",
                            "_local_idx": fallback_idx,
                            "_sents": set(child_sents)})
            best_idx = fallback_idx
        if best_idx is not None:
            child["_assigned_parent_idx"] = best_idx
        children.append(child)

    # Clean temp sets from parents
    for p in parents:
        del p["_sents"]

    return parents + children

def chunk_section_aware(sections: list[dict], **_kw) -> list[dict]:
    all_chunks = []
    for sec in sections:
        all_chunks.extend(_build_parent_child_for_section(sec))
    return all_chunks

def chunk_paragraph(sections: list[dict], **_kw) -> list[dict]:
    all_chunks = []
    for sec in sections:
        meta = _section_meta(sec)
        for para in re.split(r"\n\s*\n", sec["text"]):
            para = para.strip()
            if count_tokens(para) < MIN_CHUNK_TOKENS:
                continue
            para_sents = sentence_split(para)
            for pt in _pack_sentences_tokens(para_sents,
                                             PARENT_CHUNK_TOKENS,
                                             PARENT_OVERLAP_TOKENS):
                if count_tokens(pt) >= MIN_CHUNK_TOKENS:
                    all_chunks.append({**meta, "section_title": sec["title"],
                                       "text": pt, "level": "parent"})
            for ct in _pack_sentences_tokens(para_sents,
                                             CHILD_CHUNK_TOKENS,
                                             CHILD_OVERLAP_TOKENS):
                if count_tokens(ct) >= MIN_CHUNK_TOKENS:
                    all_chunks.append({**meta, "section_title": sec["title"],
                                       "text": ct, "level": "child"})
    return all_chunks

def chunk_lexicon(sections: list[dict], **_kw) -> list[dict]:
    result = []
    for sec in sections:
        meta = _section_meta(sec)
        lines = [l.strip() for l in sec["text"].split("\n") if l.strip()]
        cur, cur_tok = [], 0
        for line in lines:
            lt = count_tokens(line)
            if cur_tok + lt > 400 and cur:
                combined = "\n".join(cur)
                if count_tokens(combined) >= MIN_CHUNK_TOKENS:
                    result.append({**meta, "section_title": "Ontology Entry",
                                   "text": combined, "level": "child"})
                cur, cur_tok = [], 0
            cur.append(line)
            cur_tok += lt
        if cur:
            combined = "\n".join(cur)
            if count_tokens(combined) >= MIN_CHUNK_TOKENS:
                result.append({**meta, "section_title": "Ontology Entry",
                               "text": combined, "level": "child"})
    return result

CHUNK_STRATEGIES = {
    "section_aware": chunk_section_aware,
    "paragraph":     chunk_paragraph,
    "lexicon":       chunk_lexicon,
    "step_based":    chunk_section_aware,
    "fixed_semantic":chunk_section_aware,
    "recommendation": chunk_section_aware,
    "clinical_review": chunk_section_aware,
    "human_factors": chunk_section_aware,
    "dataset_paper": chunk_section_aware,
    "landmark_review": chunk_paragraph,
}

# ═══════════════════════════════════════════════════════════════════════
#  CHUNK TYPE INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def _infer_chunk_type(text: str, doc_config: dict) -> str:
    tl = text.lower()
    st = doc_config.get("source_type", "")
    family = doc_config.get("doc_family", "")
    if st == "ontology":
        return "instrument_lexicon"
    if family == "dataset_paper":
        if any(k in tl for k in ["critical view of safety", "cvs prediction", "cvs criteria"]):
            return "cvs_criteria"
        if any(k in tl for k in ["annotation", "bounding box", "segmentation mask", "dataset", "benchmark"]):
            return "instrument_lexicon"
        return "general"
    if family == "human_factors":
        if any(k in tl for k in ["optical illusion", "visual perceptual illusion", "misidentification", "misperception"]):
            return "risk_pitfall"
        return "complication_management"
    if "critical view of safety" in tl or re.search(r"\bcvs\b", tl):
        return "cvs_criteria"
    if any(k in tl for k in ["subtotal cholecystectomy", "conversion to open",
                              "bail out", "bailout", "fundus-first",
                              "fundus first", "dome-down"]):
        return "bailout_strategy"
    if any(k in tl for k in ["bile duct injury", "bile duct injuries",
                              "repair", "reconstruction",
                              "strasberg", "injury classification"]):
        return "complication_management"
    if st == "anatomy_review" or any(k in tl for k in
            ["variant", "anomaly", "aberrant", "rouviere",
             "hepatocystic triangle"]):
        if any(k in tl for k in ["variant", "anomaly", "aberrant"]):
            return "anatomy_variant"
        return "anatomy_landmark"
    if any(k in tl for k in ["technique", "procedure step",
                              "port placement", "trocar placement"]):
        return "technique_step"
    if any(k in tl for k in ["checklist", "time out", "sign in",
                              "sign out", "safety check"]):
        return "safety_check"
    return "general"

# ═══════════════════════════════════════════════════════════════════════
#  CONTEXTUALIZED TEXT
# ═══════════════════════════════════════════════════════════════════════

def _build_contextualized_text(chunk: dict, doc_config: dict) -> str:
    parts = [
        f"[Source: {doc_config['doc_title']} | "
        f"Type: {doc_config.get('source_type', 'unknown')} | "
        f"Trust: Tier {doc_config.get('trust_tier', 'B')}]"
    ]
    hp = chunk.get("heading_path", "")
    if hp:
        parts.append(f"[Section: {hp}]")
    ct = chunk.get("chunk_type", "")
    if ct and ct != "general":
        parts.append(f"[Topic: {ct.replace('_', ' ').title()}]")
    tag_parts = []
    for key, label in [("anatomy_tags", "Anatomy"), ("risk_tags", "Risk"),
                       ("phase_scope", "Phase")]:
        tags = chunk.get(key, [])
        if tags:
            tag_parts.append(f"{label}: {', '.join(tags[:5])}")
    if tag_parts:
        parts.append(f"[{' | '.join(tag_parts)}]")
    hints = doc_config.get("tags_hint", [])
    if hints:
        parts.append(f"[Focus: {', '.join(hints[:4])}]")
    ps, pe = chunk.get("page_start"), chunk.get("page_end")
    if ps and pe:
        pg = f"Page {ps}" if ps == pe else f"Pages {ps}-{pe}"
        parts.append(f"[{pg}]")
    return " ".join(parts) + "\n" + chunk["text"]

# ═══════════════════════════════════════════════════════════════════════
#  HIERARCHICAL SUMMARIES
# ═══════════════════════════════════════════════════════════════════════

def _build_section_summary(section: dict) -> dict:
    sents = sentence_split(section["text"])
    if len(sents) <= 4:
        chosen = sents
    else:
        chosen = sents[:2] + [sents[len(sents) // 2], sents[-1]]

    summary = " ".join(chosen).strip()
    if count_tokens(summary) < 60:
        for sent in sents:
            if sent in chosen:
                continue
            candidate = (summary + " " + sent).strip()
            if count_tokens(candidate) > PARENT_CHUNK_TOKENS:
                break
            summary = candidate
            if count_tokens(summary) >= 60:
                break
    while count_tokens(summary) > PARENT_CHUNK_TOKENS:
        summary = " ".join(summary.split()[:-10])
    return {**_section_meta(section), "section_title": section["title"],
            "text": summary, "level": "section_summary"}

def _build_document_summary(sections: list[dict]) -> dict:
    parts = []
    for sec in sections[:10]:
        sents = sentence_split(sec["text"])
        if sents:
            first = re.sub(rf"^\s*{re.escape(sec['title'])}\s*:\s*", "", sents[0], flags=re.IGNORECASE)
            parts.append(f"{sec['title']}: {first}")
    summary = " ".join(parts)
    while count_tokens(summary) > PARENT_CHUNK_TOKENS:
        summary = " ".join(summary.split()[:-10])
    return {
        "section_title": "Document Summary", "text": summary,
        "page_start": sections[0].get("page_start") if sections else None,
        "page_end": sections[-1].get("page_end") if sections else None,
        "heading_path": "Document Summary",
        "section_id": "DOC_SUMMARY",
        "level": "document_summary",
    }

# ═══════════════════════════════════════════════════════════════════════
#  PARENT-CHILD ID ASSIGNMENT 
# ═══════════════════════════════════════════════════════════════════════

def _assign_parent_child_ids(chunks: list[dict]):
    by_section = {}
    for c in chunks:
        by_section.setdefault(c.get("section_id"), []).append(c)

    for group in by_section.values():
        parents  = [c for c in group if c["level"] == "parent"]
        children = [c for c in group if c["level"] == "child"]
        if not parents or not children:
            # Clean temp keys
            for c in group:
                c.pop("_local_idx", None)
                c.pop("_assigned_parent_idx", None)
            continue

        parent_by_idx = {}
        for p in parents:
            idx = p.pop("_local_idx", None)
            if idx is not None:
                parent_by_idx[idx] = p

        for child in children:
            child.pop("_local_idx", None)
            assigned_idx = child.pop("_assigned_parent_idx", None)

            parent = None
            if assigned_idx is not None and assigned_idx in parent_by_idx:
                parent = parent_by_idx[assigned_idx]
            else:
                # Fallback: sentence overlap
                child_sents = set(sentence_split(child["text"]))
                best_p, best_ov = None, 0
                for p in parents:
                    ov = len(child_sents & set(sentence_split(p["text"])))
                    if ov > best_ov:
                        best_ov = ov
                        best_p = p
                parent = best_p

                if parent is None:
                    child_words = set(re.findall(r"[a-z0-9]+", child["text"].lower()))
                    best_ratio = 0.0
                    for p in parents:
                        parent_words = set(re.findall(r"[a-z0-9]+", p["text"].lower()))
                        if not child_words or not parent_words:
                            continue
                        overlap = len(child_words & parent_words) / max(len(child_words), 1)
                        if overlap > best_ratio:
                            best_ratio = overlap
                            parent = p

                if parent is None and len(parents) == 1:
                    parent = parents[0]

            if parent is not None:
                child["parent_id"] = parent["chunk_id"]
                parent.setdefault("child_ids", []).append(child["chunk_id"])

        # Sibling IDs
        for parent in parents:
            cids = parent.get("child_ids", [])
            for child in children:
                if child.get("parent_id") == parent.get("chunk_id"):
                    child["sibling_ids"] = [s for s in cids
                                            if s != child["chunk_id"]]

# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def build_chunks_for_document(doc_config: dict) -> list[dict]:
    filename = doc_config["filename"]
    doc_id   = doc_config["doc_id"]

    file_path = DOCS_RAW_DIR / filename
    if not file_path.exists():
        alt = file_path.with_suffix(".txt" if file_path.suffix == ".pdf" else ".pdf")
        if alt.exists():
            file_path = alt
            print(f"  [INFO] Using {alt.name}")
        else:
            print(f"  [!] Not found: {file_path} - skipping")
            return []

    print(f"  Extracting {file_path.name} ...")
    pages = extract_pages(file_path)
    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars < 100:
        print(f"  [!] Too short ({total_chars} chars) - skipping")
        return []
    print(f"  {len(pages)} pages, {total_chars} chars")

    sections = detect_sections_with_pages(pages)
    print(f"  {len(sections)} raw sections detected")
    sections = filter_sections_for_document(sections, doc_config)
    print(f"  {len(sections)} sections kept after document-aware filtering")
    if not sections:
        print("  [!] No usable sections after filtering - skipping")
        return []

    strategy = doc_config.get("chunk_strategy", "section_aware")
    chunk_fn = CHUNK_STRATEGIES.get(strategy, chunk_section_aware)
    raw_chunks = chunk_fn(sections)

    sec_summaries = [_build_section_summary(s) for s in sections]
    doc_summary   = _build_document_summary(sections)
    all_raw = raw_chunks + sec_summaries + [doc_summary]

    n_par = sum(1 for c in raw_chunks if c.get("level") == "parent")
    n_chi = sum(1 for c in raw_chunks if c.get("level") == "child")
    print(f"  '{strategy}' -> {n_par} parents + {n_chi} children + "
          f"{len(sec_summaries)} sec summaries + 1 doc summary")

    enriched = []
    seen_texts = set()
    for idx, rc in enumerate(all_raw):
        level = rc.get("level", "child")
        normalized_text = re.sub(r"\s+", " ", rc["text"]).strip().lower()
        if normalized_text in seen_texts:
            continue
        seen_texts.add(normalized_text)
        tags = extract_tags(rc["text"])
        chunk_type = _infer_chunk_type(rc["text"], doc_config)

        chunk = {
            "chunk_id":         f"{doc_id}_{level[:3].upper()}_{idx:04d}",
            "doc_id":           doc_id,
            "doc_title":        doc_config["doc_title"],
            "source_type":      doc_config.get("source_type", "unknown"),
            "doc_family":       doc_config.get("doc_family", "general"),
            "trust_tier":       doc_config.get("trust_tier", "B"),
            "collection":       doc_config.get("collection", "general"),
            "priority":         doc_config.get("priority", 3),
            "section_title":    rc.get("section_title", ""),
            "section_id":       rc.get("section_id", ""),
            "heading_path":     rc.get("heading_path", ""),
            "page_start":       rc.get("page_start"),
            "page_end":         rc.get("page_end"),
            "level":            level,
            "chunk_type":       chunk_type,
            "anatomy_tags":     tags["anatomy_tags"],
            "instrument_tags":  tags["instrument_tags"],
            "action_tags":      tags["action_tags"],
            "risk_tags":        tags["risk_tags"],
            "phase_scope":      tags["phase_scope"],
            "text":             rc["text"],
            "token_count":      count_tokens(rc["text"]),
        }
        for tk in ("_local_idx", "_assigned_parent_idx"):
            if tk in rc:
                chunk[tk] = rc[tk]

        chunk["contextualized_text"] = _build_contextualized_text(chunk, doc_config)
        enriched.append(chunk)

    _assign_parent_child_ids(enriched)
    return enriched

def main():
    print("=" * 60)
    print("BUILD CORPUS - SurgRAG-VQA")
    print("=" * 60)

    all_chunks = []
    stats = {"processed": 0, "skipped": 0, "total": 0,
             "by_collection": {}, "by_chunk_type": {}, "by_level": {}}

    for doc in RAG_DOCUMENTS:
        print(f"\n{'-'*60}")
        print(f"  {doc['doc_title']}")
        print(f"   Tier: {doc['trust_tier']} | Collection: {doc['collection']} | "
              f"Strategy: {doc['chunk_strategy']}")

        chunks = build_chunks_for_document(doc)
        if chunks:
            all_chunks.extend(chunks)
            stats["processed"] += 1
            stats["total"] += len(chunks)
            coll = doc["collection"]
            stats["by_collection"][coll] = stats["by_collection"].get(coll, 0) + len(chunks)
            for c in chunks:
                stats["by_chunk_type"][c["chunk_type"]] = \
                    stats["by_chunk_type"].get(c["chunk_type"], 0) + 1
                stats["by_level"][c["level"]] = \
                    stats["by_level"].get(c["level"], 0) + 1
            avg_tok = sum(c["token_count"] for c in chunks) // max(len(chunks), 1)
            print(f"  => {len(chunks)} total (avg {avg_tok} tokens)")
        else:
            stats["skipped"] += 1
            print(f"  => Skipped")

    # ── Validation ──
    print(f"\n{'-'*60}")
    print("VALIDATION:")

    total_children = sum(1 for c in all_chunks if c["level"] == "child")
    with_parent = sum(1 for c in all_chunks
                      if c["level"] == "child" and c.get("parent_id"))
    print(f"  Children with parent_id: {with_parent}/{total_children}")

    by_id = {c["chunk_id"]: c for c in all_chunks}
    mismatch = 0
    for c in all_chunks:
        if c["level"] == "child" and c.get("parent_id"):
            p = by_id.get(c["parent_id"])
            if p:
                cs = set(sentence_split(c["text"]))
                ps = set(sentence_split(p["text"]))
                if not (cs & ps):
                    mismatch += 1
    print(f"  Parent-child text mismatches: {mismatch}")

    port_fp = sum(1 for c in all_chunks
                  if any("port" in t for t in c.get("instrument_tags", []))
                  and any(w in c["text"].lower()
                          for w in ["report", "import", "support", "transport"]))
    wash_fp = sum(1 for c in all_chunks
                  if "irrigate" in c.get("action_tags", [])
                  and "washington" in c["text"].lower())
    print(f"  Tag FP - port: {port_fp}, wash->irrigate: {wash_fp}")

    bad_titles = sum(1 for c in all_chunks
                     if any(x in c.get("section_title", "")
                            for x in ["MD,", "PhD", "University", "Medical Center"]))
    print(f"  Bad section titles: {bad_titles}")

    moji = sum(1 for c in all_chunks
               if _contains_mojibake(c["text"]) or _contains_mojibake(c["contextualized_text"]))
    print(f"  Mojibake chunks: {moji}")

    intro_noise = sum(1 for c in all_chunks
                      if c.get("section_title") == "Introduction"
                      and any(x in c["text"] for x in ["Authors", "University", "Email:", "Hospital"]))
    print(f"  Intro front-matter chunks: {intro_noise}")

    # ── Write ──
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── Report ──
    print(f"\n{'='*60}")
    print("DONE - Corpus build complete")
    print(f"{'='*60}")
    print(f"  Docs: {stats['processed']} ok, {stats['skipped']} skipped")
    print(f"  Total chunks: {stats['total']}")
    print(f"  Output: {CHUNKS_FILE}")

    print(f"\n  By level:")
    for lv, n in sorted(stats["by_level"].items()):
        print(f"    {lv:25s} {n:4d}")
    print(f"\n  By collection:")
    for c, n in sorted(stats["by_collection"].items()):
        print(f"    {c:35s} {n:4d}")
    print(f"\n  By chunk type:")
    for ct, n in sorted(stats["by_chunk_type"].items()):
        print(f"    {ct:35s} {n:4d}")

    print(f"\n{'-'*60}\nSample chunks:")
    for level in ["child", "parent", "section_summary", "document_summary"]:
        for c in [x for x in all_chunks if x["level"] == level][:1]:
            print(f"\n  [{c['chunk_id']}] {c['level'].upper()} | {c['chunk_type']}")
            print(f"  Section: {c['section_title']}")
            print(f"  Pages: {c.get('page_start')}-{c.get('page_end')} | "
                  f"Tokens: {c['token_count']}")
            print(f"  Tags: anat={c['anatomy_tags'][:3]} risk={c['risk_tags'][:2]}")
            print(f"  Parent: {c.get('parent_id', '-')}")
            print(f"  Text: {c['text'][:100]}...")

if __name__ == "__main__":
    main()
