"""
build_corpus.py — Parse PDFs and create chunked corpus for RAG retrieval.

Usage:
    python scripts/build_corpus.py

Reads PDFs from docs/raw/, chunks them, and writes to docs/chunks/chunks_v1.jsonl.
Falls back to pdfplumber if pypdf extraction is poor quality.
"""

import json
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DOCS_RAW_DIR, DOCS_CHUNKS_DIR, CHUNKS_FILE,
    RAG_DOCUMENTS, CHUNK_SIZE, CHUNK_OVERLAP,
)


# ─── PDF Text Extraction ────────────────────────────────────────────

def extract_with_pypdf(pdf_path: Path) -> str:
    """Primary extractor using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_with_pdfplumber(pdf_path: Path) -> str:
    """Fallback extractor — better for scanned / complex layouts."""
    import pdfplumber
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def extract_text(pdf_path: Path) -> str:
    """
    Try pypdf first. If result looks like garbage (too short or mostly
    non-ASCII), fall back to pdfplumber.
    """
    text = extract_with_pypdf(pdf_path)

    # Heuristic: if extracted text is very short or >30 % non-ASCII → bad
    if len(text.strip()) < 200:
        print(f"  ⚠ pypdf gave only {len(text)} chars — trying pdfplumber...")
        text = extract_with_pdfplumber(pdf_path)

    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio < 0.7:
        print(f"  ⚠ pypdf text looks garbled ({ascii_ratio:.0%} ASCII) — trying pdfplumber...")
        text = extract_with_pdfplumber(pdf_path)

    return text


# ─── Chunking ────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic cleaning: collapse whitespace, remove form-feed / null."""
    text = text.replace("\x0c", "\n").replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def sentence_split(text: str) -> list[str]:
    """Split on sentence-ending punctuation, keeping headings intact."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def semantic_chunk(
    text: str,
    doc_id: str,
    doc_title: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Chunk text by sentences so we never cut mid-sentence.
    Each chunk ≈ chunk_size characters. Overlap by carrying the last
    sentence(s) of the previous chunk forward.
    """
    text = clean_text(text)
    sentences = sentence_split(text)

    chunks = []
    current_sentences: list[str] = []
    current_len = 0
    chunk_idx = 0

    for sent in sentences:
        if current_len + len(sent) > chunk_size and current_sentences:
            # Flush current chunk
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "chunk_id": f"{doc_id}_CH{chunk_idx:03d}",
                "doc_id": doc_id,
                "doc_title": doc_title,
                "text": chunk_text,
                "char_len": len(chunk_text),
            })
            chunk_idx += 1

            # Overlap: keep sentences from the tail that fit within `overlap` chars
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s)

            current_sentences = overlap_sents
            current_len = overlap_len

        current_sentences.append(sent)
        current_len += len(sent)

    # Last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append({
            "chunk_id": f"{doc_id}_CH{chunk_idx:03d}",
            "doc_id": doc_id,
            "doc_title": doc_title,
            "text": chunk_text,
            "char_len": len(chunk_text),
        })

    return chunks

# Main 

def main():
    all_chunks: list[dict] = []
    pdfs_found = False

    for filename, doc_id, doc_title in RAG_DOCUMENTS:
        pdf_path = DOCS_RAW_DIR / filename
        if not pdf_path.exists():
            print(f"⚠  Not found: {pdf_path} — skipping")
            continue

        pdfs_found = True
        print(f"Processing {pdf_path} ...")
        text = extract_text(pdf_path)
        if len(text.strip()) < 100:
            print(f"  ⚠ Extracted text too short ({len(text)} chars). "
                  "Try copy-pasting the text into a .txt file instead.")
            continue

        chunks = semantic_chunk(text, doc_id, doc_title)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks created  "
              f"(avg {sum(c['char_len'] for c in chunks)//max(len(chunks),1)} chars)")

    # Also ingest any .txt files sitting in docs/raw/
    for txt_path in sorted(DOCS_RAW_DIR.glob("*.txt")):
        print(f"Processing {txt_path} ...")
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        doc_id = txt_path.stem.upper().replace(" ", "_")[:10]
        chunks = semantic_chunk(text, doc_id, txt_path.stem)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks created")


    # Write output
    DOCS_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to:     {CHUNKS_FILE}")
    print(f"{'='*50}")

    # Quick sanity preview
    print("\nSample chunks:")
    for c in all_chunks[:3]:
        print(f"  [{c['chunk_id']}] {c['text'][:120]}...")


if __name__ == "__main__":
    main()
