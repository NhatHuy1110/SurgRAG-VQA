"""
build_corpus_v2.py — Smart corpus builder for Surgical RAG-VQA.

Fixes vs v1:
  1. All file reads use encoding="utf-8", errors="replace" (fixes Windows cp1252 crash)
  2. Aggressive junk removal (headers, footers, page numbers, copyright, URLs)
  3. Reference section truncation
  4. Section-aware chunking with heading detection
  5. Source-specific chunk strategies
  6. Rich metadata per chunk (collection, source_type, trust_tier, tags)
  7. Minimum chunk length filter

Usage:
    python scripts/build_corpus_v2.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DOCS_RAW_DIR,
    DOCS_CHUNKS_DIR,
    CHUNKS_FILE,
    RAG_DOCUMENTS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
    HEADING_PATTERNS,
    JUNK_PATTERNS,
    REFERENCE_MARKERS,
)


# ═══════════════════════════════════════════════════════════════════════
#  PDF / TEXT EXTRACTION  (encoding-safe)
# ═══════════════════════════════════════════════════════════════════════

def extract_with_pypdf(pdf_path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_with_pdfplumber(pdf_path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def extract_text(file_path: Path) -> str:
    """Extract text from PDF or TXT. Always UTF-8 safe."""
    # ── TXT files ──
    if file_path.suffix.lower() in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    # ── PDF files ──
    text = ""
    try:
        text = extract_with_pypdf(file_path)
    except Exception as e:
        print(f"  ⚠ pypdf failed: {e}")

    # Quality check: too short → fallback
    if len(text.strip()) < 200:
        print(f"  ⚠ pypdf gave only {len(text)} chars — trying pdfplumber...")
        try:
            text = extract_with_pdfplumber(file_path)
        except Exception as e:
            print(f"  ⚠ pdfplumber also failed: {e}")

    # Quality check: garbled text → fallback
    if text:
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        if ascii_ratio < 0.7:
            print(f"  ⚠ pypdf text looks garbled ({ascii_ratio:.0%} ASCII) — trying pdfplumber...")
            try:
                text = extract_with_pdfplumber(file_path)
            except Exception as e:
                print(f"  ⚠ pdfplumber also failed: {e}")

    return text


# ═══════════════════════════════════════════════════════════════════════
#  TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════

_junk_re = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in JUNK_PATTERNS]
_ref_re  = [re.compile(p, re.MULTILINE) for p in REFERENCE_MARKERS]
_heading_re = [re.compile(p, re.MULTILINE) for p in HEADING_PATTERNS]


def remove_junk_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if any(pat.match(line.strip()) for pat in _junk_re):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def truncate_at_references(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        for pat in _ref_re:
            if pat.match(line.strip()):
                truncated = "\n".join(lines[:i])
                if len(truncated.strip()) > 500:
                    return truncated
    return text


def clean_text(text: str) -> str:
    text = text.replace("\x0c", "\n").replace("\x00", "")
    text = remove_junk_lines(text)
    text = truncate_at_references(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════
#  SECTION DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_sections(text: str) -> list[dict]:
    lines = text.split("\n")
    sections = []
    current_title = "Introduction"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        is_heading = False

        if stripped and len(stripped) < 200:
            for pat in _heading_re:
                if pat.match(stripped):
                    is_heading = True
                    break

        if is_heading and current_lines:
            section_text = "\n".join(current_lines).strip()
            if len(section_text) > MIN_CHUNK_LENGTH:
                sections.append({"title": current_title, "text": section_text})
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if len(section_text) > MIN_CHUNK_LENGTH:
            sections.append({"title": current_title, "text": section_text})

    if not sections:
        sections = [{"title": "Full Document", "text": text}]

    return sections


# ═══════════════════════════════════════════════════════════════════════
#  SENTENCE SPLITTING (improved)
# ═══════════════════════════════════════════════════════════════════════

def sentence_split(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    sentences = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Protect abbreviations
        protected = para
        abbrevs = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "vs.", "etc.",
                    "i.e.", "e.g.", "Fig.", "fig.", "Vol.", "vol.",
                    "No.", "no.", "Ref.", "ref.", "approx.", "ca.",
                    "et al.", "Sr.", "Jr."]
        for abbr in abbrevs:
            protected = protected.replace(abbr, abbr.replace(".", "§"))

        parts = re.split(r'(?<=[.!?])\s+', protected)
        for part in parts:
            restored = part.replace("§", ".").strip()
            if restored:
                sentences.append(restored)

    return sentences


# ═══════════════════════════════════════════════════════════════════════
#  METADATA ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════

ANATOMY_KEYWORDS = {
    "gallbladder", "cystic duct", "common bile duct", "common hepatic duct",
    "hepatic duct", "cystic artery", "hepatic artery", "right hepatic artery",
    "portal vein", "liver bed", "gallbladder fossa", "fundus", "infundibulum",
    "hartmann pouch", "gallbladder neck", "serosa", "peritoneum",
    "hepatocystic triangle", "calot triangle", "cystic plate",
    "rouviere sulcus", "rouviere's sulcus", "segment iv", "segment v",
    "omentum", "duodenum", "colon", "bile duct", "biliary",
    "ampulla", "sphincter of oddi",
}

INSTRUMENT_KEYWORDS = {
    "grasper", "dissector", "maryland dissector", "hook", "electrocautery",
    "monopolar", "bipolar", "clip applier", "clip", "scissors",
    "irrigator", "suction", "trocar", "port", "camera", "laparoscope",
    "endoscope", "retractor", "specimen bag", "endo bag",
    "cholangiography catheter", "drain",
}

ACTION_KEYWORDS = {
    "dissect", "dissection", "retract", "retraction", "clip", "clipping",
    "divide", "division", "cauterize", "coagulate", "irrigate", "suction",
    "grasp", "grasping", "expose", "exposure", "peel", "strip",
    "incise", "incision", "extract", "extraction",
}

RISK_KEYWORDS = {
    "bile duct injury", "bdi", "bleeding", "hemorrhage", "hemostasis",
    "bile leak", "bile spillage", "perforation", "thermal injury",
    "misidentification", "wrong structure", "conversion",
    "inflammation", "fibrosis", "adhesion", "cholecystitis",
    "empyema", "gangrene", "necrosis",
}

PHASE_KEYWORDS = {
    "setup":                  ["port placement", "trocar", "insufflation", "pneumoperitoneum"],
    "exposure":               ["retraction", "expose", "visualization", "camera position"],
    "dissection":             ["calot", "hepatocystic", "peritoneal", "dissect",
                               "critical view of safety", "cvs"],
    "clipping_division":      ["clip", "divide", "cystic duct", "cystic artery",
                               "ligate", "transect"],
    "gallbladder_separation": ["liver bed", "gallbladder fossa", "peel",
                               "electrocautery", "separation"],
    "extraction":             ["specimen bag", "extract", "retrieve", "port site"],
    "hemostasis_inspection":  ["hemostasis", "inspect", "bile leak", "drain"],
}


def extract_tags(text: str) -> dict:
    text_lower = text.lower()
    anatomy     = sorted(kw for kw in ANATOMY_KEYWORDS if kw in text_lower)
    instruments = sorted(kw for kw in INSTRUMENT_KEYWORDS if kw in text_lower)
    actions     = sorted(kw for kw in ACTION_KEYWORDS if kw in text_lower)
    risks       = sorted(kw for kw in RISK_KEYWORDS if kw in text_lower)

    phases = []
    for phase_name, keywords in PHASE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            phases.append(phase_name)

    return {
        "anatomy_tags":    anatomy,
        "instrument_tags": instruments,
        "action_tags":     actions,
        "risk_tags":       risks,
        "phase_scope":     phases,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CHUNKING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

def _pack_sentences(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s)
            current = overlap_sents
            current_len = overlap_len

        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_section_aware(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    sections = detect_sections(text)
    result = []
    for section in sections:
        sentences = sentence_split(section["text"])
        if not sentences:
            continue
        for ct in _pack_sentences(sentences, chunk_size, overlap):
            if len(ct.strip()) >= MIN_CHUNK_LENGTH:
                result.append({"section_title": section["title"], "text": ct.strip()})
    return result


def chunk_step_based(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    step_pattern = re.compile(
        r"(?:^|\n)(?=\s*(?:Step\s+\d|Phase\s+\d|\d+[\.\)]\s+[A-Z]|"
        r"Technique:|Procedure:|Equipment:|Indications:|Contraindications:|"
        r"Preparation:|Anatomy:|Complications?))",
        re.MULTILINE,
    )
    parts = step_pattern.split(text)
    if len(parts) <= 1:
        return chunk_section_aware(text, chunk_size, overlap)

    result = []
    for part in parts:
        part = part.strip()
        if len(part) < MIN_CHUNK_LENGTH:
            continue
        first_line = part.split("\n")[0].strip()
        step_title = first_line[:100] if len(first_line) < 120 else "Procedure Step"
        sentences = sentence_split(part)
        for ct in _pack_sentences(sentences, chunk_size, overlap):
            if len(ct.strip()) >= MIN_CHUNK_LENGTH:
                result.append({"section_title": step_title, "text": ct.strip()})
    return result


def chunk_paragraph(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    paragraphs = re.split(r"\n\s*\n", text)
    result = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < MIN_CHUNK_LENGTH:
            continue
        if len(para) <= chunk_size:
            result.append({"section_title": para.split("\n")[0][:80], "text": para})
        else:
            sentences = sentence_split(para)
            for ct in _pack_sentences(sentences, chunk_size, overlap):
                if len(ct.strip()) >= MIN_CHUNK_LENGTH:
                    result.append({"section_title": para.split("\n")[0][:80], "text": ct.strip()})
    return result


def chunk_lexicon(text, chunk_size=400, **_kw):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    result = []
    current = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > chunk_size and current:
            combined = "\n".join(current)
            if len(combined) >= MIN_CHUNK_LENGTH:
                result.append({"section_title": "Ontology Entry", "text": combined})
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)
    if current:
        combined = "\n".join(current)
        if len(combined) >= MIN_CHUNK_LENGTH:
            result.append({"section_title": "Ontology Entry", "text": combined})
    return result


CHUNK_STRATEGIES = {
    "section_aware":  chunk_section_aware,
    "step_based":     chunk_step_based,
    "paragraph":      chunk_paragraph,
    "lexicon":        chunk_lexicon,
    "fixed_semantic": chunk_section_aware,
}


# ═══════════════════════════════════════════════════════════════════════
#  CHUNK TYPE INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def _infer_chunk_type(text: str, doc_config: dict) -> str:
    text_lower = text.lower()
    source_type = doc_config.get("source_type", "")

    if source_type == "ontology":
        if any(kw in text_lower for kw in ["instrument", "grasper", "hook", "clipper"]):
            return "instrument_lexicon"
        return "instrument_lexicon"

    if "critical view of safety" in text_lower or "cvs" in text_lower:
        return "cvs_criteria"

    if any(kw in text_lower for kw in [
        "subtotal cholecystectomy", "conversion", "bail out", "bailout",
        "fundus-first", "fundus first", "dome-down",
    ]):
        return "bailout_strategy"

    if any(kw in text_lower for kw in [
        "bile duct injury", "bdi", "repair", "reconstruction",
        "strasberg classification", "injury classification",
    ]):
        return "complication_management"

    if source_type == "anatomy_review" or any(kw in text_lower for kw in [
        "anatomy", "variant", "anomaly", "aberrant", "rouviere",
        "hepatocystic triangle", "calot",
    ]):
        if any(kw in text_lower for kw in ["variant", "anomaly", "aberrant"]):
            return "anatomy_variant"
        return "anatomy_landmark"

    if source_type == "operative_technique" or any(kw in text_lower for kw in [
        "step", "technique", "procedure", "port placement", "trocar",
        "retraction", "dissection", "clipping", "extraction",
    ]):
        return "technique_step"

    if any(kw in text_lower for kw in [
        "checklist", "time out", "sign in", "sign out", "safety check",
    ]):
        return "safety_check"

    return "general"


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def build_chunks_for_document(doc_config: dict) -> list[dict]:
    filename = doc_config["filename"]
    doc_id = doc_config["doc_id"]
    doc_title = doc_config["doc_title"]

    # Find file — try original extension, then alternate
    file_path = DOCS_RAW_DIR / filename
    if not file_path.exists():
        alt_ext = ".txt" if file_path.suffix == ".pdf" else ".pdf"
        alt_path = file_path.with_suffix(alt_ext)
        if alt_path.exists():
            file_path = alt_path
            print(f"  i Using {alt_path.name} instead of {filename}")
        else:
            print(f"  [!] Not found: {file_path} — skipping")
            return []

    print(f"  Extracting text from {file_path.name} ...")
    raw_text = extract_text(file_path)
    if len(raw_text.strip()) < 100:
        print(f"  [!] Extracted text too short ({len(raw_text)} chars) — skipping")
        return []

    # Clean
    text = clean_text(raw_text)
    removed_pct = 100 - len(text) * 100 // max(len(raw_text), 1)
    print(f"  Cleaned: {len(raw_text)} -> {len(text)} chars ({removed_pct}% removed)")

    # Chunk
    strategy_name = doc_config.get("chunk_strategy", "section_aware")
    chunk_size = doc_config.get("chunk_size", DEFAULT_CHUNK_SIZE)
    chunk_fn = CHUNK_STRATEGIES.get(strategy_name, chunk_section_aware)

    raw_chunks = chunk_fn(text, chunk_size=chunk_size, overlap=DEFAULT_CHUNK_OVERLAP)
    print(f"  Strategy '{strategy_name}' -> {len(raw_chunks)} raw chunks")

    # Enrich metadata
    enriched = []
    for idx, rc in enumerate(raw_chunks):
        tags = extract_tags(rc["text"])
        chunk = {
            "chunk_id":        f"{doc_id}_CH{idx:03d}",
            "doc_id":          doc_id,
            "doc_title":       doc_title,
            "source_type":     doc_config.get("source_type", "unknown"),
            "trust_tier":      doc_config.get("trust_tier", "B"),
            "collection":      doc_config.get("collection", "general"),
            "priority":        doc_config.get("priority", 3),
            "section_title":   rc.get("section_title", ""),
            "chunk_type":      _infer_chunk_type(rc["text"], doc_config),
            "anatomy_tags":    tags["anatomy_tags"],
            "instrument_tags": tags["instrument_tags"],
            "action_tags":     tags["action_tags"],
            "risk_tags":       tags["risk_tags"],
            "phase_scope":     tags["phase_scope"],
            "text":            rc["text"],
            "char_len":        len(rc["text"]),
        }
        enriched.append(chunk)

    return enriched


def main():
    print("=" * 60)
    print("BUILD CORPUS V2")
    print("=" * 60)

    all_chunks = []
    stats = {
        "processed": 0, "skipped": 0, "total": 0,
        "by_collection": {}, "by_chunk_type": {}, "by_source": {},
    }

    for doc in RAG_DOCUMENTS:
        print(f"\n{'─'*60}")
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
                ct = c["chunk_type"]
                stats["by_chunk_type"][ct] = stats["by_chunk_type"].get(ct, 0) + 1
                st = c["source_type"]
                stats["by_source"][st] = stats["by_source"].get(st, 0) + 1

            avg_len = sum(c["char_len"] for c in chunks) // max(len(chunks), 1)
            print(f"  => {len(chunks)} chunks (avg {avg_len} chars)")
        else:
            stats["skipped"] += 1
            print(f"  => Skipped")

    # Write
    DOCS_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Documents: {stats['processed']} processed, {stats['skipped']} skipped")
    print(f"  Total chunks: {stats['total']}")
    print(f"  Output: {CHUNKS_FILE}")

    print(f"\n  By collection:")
    for coll, cnt in sorted(stats["by_collection"].items()):
        print(f"    {coll:35s} {cnt:4d}")

    print(f"\n  By chunk type:")
    for ct, cnt in sorted(stats["by_chunk_type"].items()):
        print(f"    {ct:35s} {cnt:4d}")

    # Sample
    print(f"\n{'─'*60}")
    print("Sample chunks:")
    for c in all_chunks[:3]:
        print(f"\n  [{c['chunk_id']}] {c['doc_title']}")
        print(f"  Type: {c['chunk_type']} | Coll: {c['collection']}")
        print(f"  Section: {c['section_title']}")
        print(f"  Anat: {c['anatomy_tags'][:3]} | Inst: {c['instrument_tags'][:3]} | "
              f"Risk: {c['risk_tags'][:3]}")
        print(f"  Text: {c['text'][:120]}...")


if __name__ == "__main__":
    main()