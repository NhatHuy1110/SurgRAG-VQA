"""
retrieval_v3.py - Retrieval v3 for SurgRAG-VQA.

Core changes over v2:
  - defaults to chunks_v3.jsonl
  - indexes contextualized_text instead of raw text only
  - retrieves on child chunks, expands to parent context for evidence packaging
  - question-conditioned query building with question_type / extra terms
  - field-aware BM25 + dense retrieval + reciprocal rank fusion
  - adaptive selection with diversity filtering
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CHUNKS_FILE,
    DENSE_MODEL_NAME,
    USE_RERANKER,
    RERANKER_MODEL_NAME,
    RERANK_TOP_N,
    HYBRID_ALPHA,
    RETRIEVAL_TOP_K,
    RETRIEVAL_MODE,
    HF_TOKEN,
    HF_CACHE_DIR,
    HF_LOCAL_FILES_ONLY,
    COLLECTION_PRIORITY,
    RETRIEVAL_EVAL_FILE,
    QUESTIONS_FILE,
)


FIELD_WEIGHTS = {
    "contextualized_text": 1.00,
    "doc_title": 0.40,
    "section_title": 0.65,
    "chunk_type": 0.75,
    "anatomy_tags": 0.80,
    "risk_tags": 0.65,
    "phase_scope": 0.60,
    "instrument_tags": 0.45,
    "action_tags": 0.45,
}

TRUST_TIER_BOOST = {
    "A": 1.08,
    "B": 1.03,
    "C": 0.99,
    "D": 0.94,
}

QUESTION_TYPE_HINTS = {
    "recognition": {
        "collections": ["visual_ontology", "biliary_anatomy_landmarks"],
        "chunk_types": ["instrument_lexicon", "anatomy_landmark", "anatomy_variant"],
        "extra_terms": ["anatomy", "structure", "operative field", "landmark", "visualized"],
    },
    "anatomy_landmark": {
        "collections": ["biliary_anatomy_landmarks", "visual_ontology", "safe_chole_guideline"],
        "chunk_types": ["anatomy_landmark", "anatomy_variant", "cvs_criteria"],
        "extra_terms": ["landmark", "hepatocystic triangle", "rouviere sulcus", "cystic duct", "cystic artery"],
    },
    "workflow_phase": {
        "collections": ["safe_chole_guideline", "general_or_safety", "visual_ontology"],
        "chunk_types": ["technique_step", "safety_check", "general"],
        "extra_terms": ["workflow", "operative step", "surgical phase", "next step"],
    },
    "safety_verification": {
        "collections": ["safe_chole_guideline", "complication_management", "biliary_anatomy_landmarks"],
        "chunk_types": ["cvs_criteria", "safety_check", "complication_management", "technique_step"],
        "extra_terms": ["safety", "verification", "critical view of safety", "two structures", "cystic plate"],
    },
    "risk_pitfall": {
        "collections": ["complication_management", "safe_chole_guideline", "biliary_anatomy_landmarks"],
        "chunk_types": ["complication_management", "bailout_strategy", "cvs_criteria", "anatomy_variant"],
        "extra_terms": ["pitfall", "risk", "injury", "misidentification", "bailout", "unclear anatomy"],
    },
}

TERM_EXPANSIONS = {
    "cvs": ["critical view of safety", "two structures", "cystic plate", "hepatocystic triangle"],
    "critical view": ["critical view of safety", "cvs", "cystic duct", "cystic artery", "cystic plate"],
    "calot": ["hepatocystic triangle", "calot triangle"],
    "bleeding": ["hemorrhage", "hemostasis", "vascular injury"],
    "bile duct injury": ["bdi", "misidentification", "bile leak", "common bile duct"],
    "subtotal": ["subtotal cholecystectomy", "bailout strategy", "fundus-first"],
    "fundus first": ["fundus-first", "dome-down", "bailout strategy"],
    "time out": ["checklist", "sign in", "sign out", "team communication"],
}

CLASS_TERM_MAP = {
    "liver_ligament": ["liver", "liver bed", "hepatobiliary tissue"],
    "hepatic_vein": ["hepatic vein", "vascular structure", "liver"],
    "gallbladder": ["gallbladder", "fundus", "infundibulum"],
    "cystic_duct": ["cystic duct"],
    "cystic_artery": ["cystic artery"],
    "grasper": ["grasper", "retraction"],
    "hook": ["hook electrocautery", "dissection"],
}

LOW_VALUE_SECTION_PATTERNS = [
    re.compile(r"\bdiscussion\b", re.IGNORECASE),
    re.compile(r"\bcase report\b", re.IGNORECASE),
    re.compile(r"\bresults?\b", re.IGNORECASE),
    re.compile(r"\bmethods?\b", re.IGNORECASE),
    re.compile(r"\bethics?\b", re.IGNORECASE),
    re.compile(r"\bstudy participants?\b", re.IGNORECASE),
]

LOW_VALUE_TEXT_PATTERNS = [
    re.compile(r"\btable\s+[ivx0-9]+\b", re.IGNORECASE),
    re.compile(r"\bexamples of annotated data\b", re.IGNORECASE),
    re.compile(r"\bcorresponding class names\b", re.IGNORECASE),
    re.compile(r"\bselected a subset of the highly related videos\b", re.IGNORECASE),
    re.compile(r"\ball rights reserved\b", re.IGNORECASE),
    re.compile(r"\barticle published online\b", re.IGNORECASE),
    re.compile(r"\bdoi\b", re.IGNORECASE),
    re.compile(r"\bet al\.", re.IGNORECASE),
    re.compile(r"\b[a-z][a-z]+ endosc\.\s+\d{4}", re.IGNORECASE),
]

INSTRUMENT_QUERY_TERMS = {
    "instrument",
    "instruments",
    "tool",
    "tools",
    "grasper",
    "hook",
    "clip",
    "clipper",
    "electrocautery",
    "scissors",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", (text or "").lower())


def _unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        norm = item.strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        ordered.append(item.strip())
    return ordered


def _join_tags(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value or "")


def _safe_console_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


class SurgicalRetrieverV2:

    def __init__(
        self,
        chunks_path: Optional[str] = None,
        dense_model: str = DENSE_MODEL_NAME,
        alpha: float = HYBRID_ALPHA,
    ):
        default_chunks = CHUNKS_FILE
        chunks_path = chunks_path or str(default_chunks)

        self.chunks: list[dict] = []
        with open(chunks_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        if not self.chunks:
            raise ValueError("Chunk file is empty. Run build_corpus_v3.py first.")

        self.alpha = alpha
        self.retrieval_mode = RETRIEVAL_MODE
        self.encoder = None
        self.faiss_index = None
        self.use_reranker = USE_RERANKER
        self.reranker_model_name = RERANKER_MODEL_NAME
        self.rerank_top_n = max(RERANK_TOP_N, RETRIEVAL_TOP_K)
        self.reranker_tokenizer = None
        self.reranker_model = None
        self.reranker_device = "cpu"

        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.parent_by_id = {
            c["chunk_id"]: c for c in self.chunks if c.get("level") == "parent"
        }
        self.index_chunks = [c for c in self.chunks if c.get("level") == "child"]
        if not self.index_chunks:
            self.index_chunks = [
                c for c in self.chunks
                if c.get("level") not in {"document_summary", "section_summary"}
            ]

        for chunk in self.chunks:
            chunk["_retrieval_text"] = chunk.get("contextualized_text") or chunk.get("text", "")

        print(f"[RetrieverV3] Loaded {len(self.chunks)} chunks from {chunks_path}")
        print(f"[RetrieverV3] Indexing {len(self.index_chunks)} retrieval units (child-first)")

        collections = {}
        levels = {}
        for c in self.chunks:
            coll = c.get("collection", "unknown")
            lvl = c.get("level", "unknown")
            collections[coll] = collections.get(coll, 0) + 1
            levels[lvl] = levels.get(lvl, 0) + 1
        for lvl, count in sorted(levels.items()):
            print(f"  level={lvl}: {count}")
        for coll, count in sorted(collections.items()):
            print(f"  collection={coll}: {count}")

        self._build_bm25()
        self._load_reranker()

        if self.retrieval_mode == "bm25_only":
            print("[RetrieverV3] Dense retrieval disabled by RETRIEVAL_MODE=bm25_only")
            return

        self._build_dense(dense_model)

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        self.bm25_fields = {}
        for field in FIELD_WEIGHTS:
            docs = []
            for chunk in self.index_chunks:
                if field == "contextualized_text":
                    value = chunk.get("_retrieval_text", "")
                elif field in {"anatomy_tags", "risk_tags", "phase_scope", "instrument_tags", "action_tags"}:
                    value = _join_tags(chunk.get(field))
                else:
                    value = chunk.get(field, "")
                docs.append(value)
            self.bm25_fields[field] = BM25Okapi([_tokenize(doc) for doc in docs])
        print("[RetrieverV3] Field-aware BM25 indexes built")

    def _build_dense(self, dense_model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            print(f"[RetrieverV3] Loading dense model: {dense_model} ...")
            t0 = time.time()
            self.encoder = SentenceTransformer(
                dense_model,
                cache_folder=HF_CACHE_DIR or None,
                local_files_only=HF_LOCAL_FILES_ONLY,
                token=HF_TOKEN or None,
            )
            embeddings = self.encoder.encode(
                [c["_retrieval_text"] for c in self.index_chunks],
                show_progress_bar=True,
                batch_size=64,
                normalize_embeddings=True,
            )
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(embeddings.astype("float32"))
            print(f"[RetrieverV3] Dense index built ({dim}d, {time.time()-t0:.1f}s)")
        except Exception as e:
            self.retrieval_mode = "bm25_only"
            self.encoder = None
            self.faiss_index = None
            print(f"[RetrieverV3] Dense retrieval unavailable, falling back to BM25-only: {e}")

    def _load_reranker(self) -> None:
        if not self.use_reranker:
            print("[RetrieverV3] Reranker disabled")
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[RetrieverV3] Loading reranker: {self.reranker_model_name} ...")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                self.reranker_model_name,
                cache_dir=HF_CACHE_DIR or None,
                local_files_only=HF_LOCAL_FILES_ONLY,
                token=HF_TOKEN or None,
            )
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name,
                cache_dir=HF_CACHE_DIR or None,
                local_files_only=HF_LOCAL_FILES_ONLY,
                token=HF_TOKEN or None,
            )
            self.reranker_model.to(self.reranker_device)
            self.reranker_model.eval()
            print(f"[RetrieverV3] Reranker ready on {self.reranker_device}")
        except Exception as e:
            self.use_reranker = False
            self.reranker_tokenizer = None
            self.reranker_model = None
            print(f"[RetrieverV3] Reranker unavailable, continuing without rerank: {e}")

    def _normalize_query_terms(self, question: str, extra_terms: Optional[list[str]] = None) -> list[str]:
        terms = [question]
        ql = question.lower()
        for trigger, expansions in TERM_EXPANSIONS.items():
            if trigger in ql:
                terms.extend(expansions)
        if extra_terms:
            terms.extend(extra_terms)
        return _unique_keep_order(terms)

    def _question_type_hint(self, question_type: Optional[str]) -> dict:
        return QUESTION_TYPE_HINTS.get(question_type or "", {})

    def build_query_bundle(
        self,
        question: str,
        question_type: Optional[str] = None,
        extra_terms: Optional[list[str]] = None,
        visual_terms: Optional[list[str]] = None,
        classes_detected: Optional[dict] = None,
    ) -> dict:
        hint = self._question_type_hint(question_type)
        all_extra_terms = list(hint.get("extra_terms", []))
        if extra_terms:
            all_extra_terms.extend(extra_terms)
        if visual_terms:
            all_extra_terms.extend(visual_terms)
        if classes_detected:
            sorted_classes = sorted(
                classes_detected.items(),
                key=lambda kv: float(kv[1] or 0.0),
                reverse=True,
            )
            for name, score in classes_detected.items():
                if score and float(score) >= 0.05:
                    all_extra_terms.append(name.replace("_", " "))
                    all_extra_terms.extend(CLASS_TERM_MAP.get(name, []))
            if question_type in {"recognition", "anatomy_landmark"}:
                for name, score in sorted_classes[:2]:
                    if score and float(score) >= 0.05:
                        all_extra_terms.extend(CLASS_TERM_MAP.get(name, []))

        terms = self._normalize_query_terms(question, all_extra_terms)
        sparse_query = " ".join(terms)
        dense_query = sparse_query

        return {
            "question": question,
            "question_type": question_type,
            "terms": terms,
            "sparse_query": sparse_query,
            "dense_query": dense_query,
            "preferred_collections": hint.get("collections", []),
            "preferred_chunk_types": hint.get("chunk_types", []),
        }

    def rerank_candidates(
        self,
        query: str,
        candidates: list[tuple[dict, float]],
        top_k: int,
    ) -> list[tuple[dict, float]]:
        if (
            not self.use_reranker
            or self.reranker_model is None
            or self.reranker_tokenizer is None
            or not candidates
        ):
            return candidates[:top_k]

        import torch

        pool = candidates[: min(len(candidates), max(top_k, self.rerank_top_n))]
        pairs = [
            (
                query,
                chunk.get("evidence_text") or chunk.get("_retrieval_text") or chunk.get("text", ""),
            )
            for chunk, _ in pool
        ]

        with torch.no_grad():
            encoded = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {
                k: (v.to(self.reranker_device) if hasattr(v, "to") else v)
                for k, v in encoded.items()
            }
            outputs = self.reranker_model(**encoded)
            logits = outputs.logits
            scores = logits[:, 0] if logits.ndim > 1 else logits
            scores = scores.detach().cpu().float().tolist()

        reranked = [(chunk, float(score)) for (chunk, _), score in zip(pool, scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def retrieve_bm25(self, query_bundle: dict, top_k: int = RETRIEVAL_TOP_K):
        tokens = _tokenize(query_bundle["sparse_query"])
        if not tokens:
            return []

        scores = np.zeros(len(self.index_chunks), dtype=float)
        for field, weight in FIELD_WEIGHTS.items():
            scores += weight * np.array(self.bm25_fields[field].get_scores(tokens), dtype=float)

        preferred_collections = set(query_bundle.get("preferred_collections", []))
        preferred_chunk_types = set(query_bundle.get("preferred_chunk_types", []))
        query_terms = {t.lower() for t in query_bundle.get("terms", [])}

        for idx, chunk in enumerate(self.index_chunks):
            bonus = 0.0
            if chunk.get("collection") in preferred_collections:
                bonus += 0.45
            if chunk.get("chunk_type") in preferred_chunk_types:
                bonus += 0.55
            if query_terms.intersection({t.lower() for t in chunk.get("anatomy_tags", [])}):
                bonus += 0.30
            if query_terms.intersection({t.lower() for t in chunk.get("risk_tags", [])}):
                bonus += 0.25
            if query_terms.intersection({t.lower() for t in chunk.get("phase_scope", [])}):
                bonus += 0.20
            scores[idx] += bonus

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.index_chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def retrieve_dense(self, query_bundle: dict, top_k: int = RETRIEVAL_TOP_K):
        if self.encoder is None or self.faiss_index is None:
            return []
        q_emb = self.encoder.encode(
            [query_bundle["dense_query"]],
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.faiss_index.search(q_emb, top_k)
        return [
            (self.index_chunks[indices[0][i]], float(scores[0][i]))
            for i in range(top_k)
            if 0 <= indices[0][i] < len(self.index_chunks)
        ]

    def _fuse_rrf(self, *result_sets: list[tuple[dict, float]], k: int = 60) -> list[tuple[dict, float]]:
        combined: dict[str, dict] = {}
        for results in result_sets:
            for rank, (chunk, _) in enumerate(results, 1):
                cid = chunk["chunk_id"]
                combined.setdefault(cid, {"chunk": chunk, "score": 0.0})
                combined[cid]["score"] += 1.0 / (k + rank)
        return [
            (entry["chunk"], float(entry["score"]))
            for entry in sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        ]

    def _apply_priors(
        self,
        ranked: list[tuple[dict, float]],
        query_bundle: dict,
        filter_collections: Optional[list[str]] = None,
        filter_chunk_types: Optional[list[str]] = None,
        boost_collections: bool = True,
    ) -> list[tuple[dict, float]]:
        filtered = []
        preferred_collections = set(query_bundle.get("preferred_collections", []))
        preferred_chunk_types = set(query_bundle.get("preferred_chunk_types", []))
        question_type = query_bundle.get("question_type")
        query_terms = {term.lower() for term in query_bundle.get("terms", [])}

        for chunk, score in ranked:
            if filter_collections and chunk.get("collection") not in filter_collections:
                continue
            if filter_chunk_types and chunk.get("chunk_type") not in filter_chunk_types:
                continue

            boosted = score
            if boost_collections:
                boosted *= COLLECTION_PRIORITY.get(chunk.get("collection", "general"), 0.7)
            boosted *= TRUST_TIER_BOOST.get(chunk.get("trust_tier", "C"), 1.0)

            if preferred_collections:
                boosted *= 1.18 if chunk.get("collection") in preferred_collections else 0.90
            if preferred_chunk_types:
                boosted *= 1.22 if chunk.get("chunk_type") in preferred_chunk_types else 0.85

            section_title = chunk.get("section_title", "") or ""
            retrieval_text = chunk.get("_retrieval_text", "") or chunk.get("text", "")
            if any(p.search(section_title) for p in LOW_VALUE_SECTION_PATTERNS):
                boosted *= 0.60
            if any(p.search(retrieval_text[:500]) for p in LOW_VALUE_SECTION_PATTERNS):
                boosted *= 0.72
            if any(p.search(retrieval_text[:700]) for p in LOW_VALUE_TEXT_PATTERNS):
                boosted *= 0.52
            if (
                chunk.get("chunk_type") == "instrument_lexicon"
                and query_terms.isdisjoint(INSTRUMENT_QUERY_TERMS)
            ):
                boosted *= 0.58

            if question_type == "recognition":
                if chunk.get("collection") == "visual_ontology":
                    boosted *= 1.45
                elif chunk.get("collection") == "biliary_anatomy_landmarks":
                    boosted *= 1.18
                else:
                    boosted *= 0.72
                if chunk.get("chunk_type") == "general":
                    boosted *= 0.55
                if chunk.get("chunk_type") == "instrument_lexicon":
                    boosted *= 0.62

            elif question_type == "anatomy_landmark":
                if chunk.get("collection") == "biliary_anatomy_landmarks":
                    boosted *= 1.22
                if chunk.get("chunk_type") == "general":
                    boosted *= 0.70
                if chunk.get("chunk_type") == "instrument_lexicon":
                    boosted *= 0.60

            elif question_type == "safety_verification":
                if chunk.get("collection") in {"safe_chole_guideline", "complication_management"}:
                    boosted *= 1.22
                elif chunk.get("collection") == "biliary_anatomy_landmarks":
                    boosted *= 0.92
                if chunk.get("chunk_type") == "general":
                    boosted *= 0.70

            elif question_type == "risk_pitfall":
                if chunk.get("collection") in {"complication_management", "safe_chole_guideline"}:
                    boosted *= 1.18
                if chunk.get("chunk_type") == "general":
                    boosted *= 0.68

            filtered.append((chunk, float(boosted)))

        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered

    def _adaptive_select(
        self,
        ranked: list[tuple[dict, float]],
        top_k: int,
    ) -> list[tuple[dict, float]]:
        if not ranked:
            return []

        selected = []
        seen_parent_keys = set()
        seen_sections = set()
        top_score = ranked[0][1] or 1.0

        for chunk, score in ranked:
            rel = score / top_score if top_score else 0.0
            if selected and len(selected) >= max(2, min(3, top_k)):
                prev = selected[-1][1] or 1.0
                if rel < 0.52:
                    break
                if prev > 0 and (score / prev) < 0.62 and rel < 0.82:
                    break

            parent_key = chunk.get("parent_id") or chunk.get("section_id") or chunk["chunk_id"]
            section_key = (chunk.get("doc_id"), chunk.get("section_id"))

            if parent_key in seen_parent_keys:
                continue
            if section_key in seen_sections and len(selected) >= max(1, top_k // 2):
                continue

            selected.append((chunk, score))
            seen_parent_keys.add(parent_key)
            seen_sections.add(section_key)
            if len(selected) >= top_k:
                break

        if len(selected) < min(top_k, 2):
            seen_ids = {c["chunk_id"] for c, _ in selected}
            for chunk, score in ranked:
                if chunk["chunk_id"] in seen_ids:
                    continue
                selected.append((chunk, score))
                seen_ids.add(chunk["chunk_id"])
                if len(selected) >= top_k:
                    break
        return selected[:top_k]

    def _package_candidate(self, chunk: dict, score: float, expand_parents: bool = True) -> dict:
        packaged = dict(chunk)
        packaged["retrieval_score"] = float(score)
        packaged["matched_chunk_id"] = chunk["chunk_id"]
        packaged["matched_level"] = chunk.get("level")
        packaged["matched_text"] = chunk.get("text", "")

        parent = self.parent_by_id.get(chunk.get("parent_id")) if expand_parents else None
        evidence = parent or chunk

        packaged["evidence_chunk_id"] = evidence["chunk_id"]
        packaged["evidence_level"] = evidence.get("level")
        packaged["evidence_text"] = evidence.get("contextualized_text") or evidence.get("text", "")
        packaged["evidence_raw_text"] = evidence.get("text", "")
        packaged["text"] = packaged["evidence_text"]
        packaged["evidence_section_title"] = evidence.get("section_title", chunk.get("section_title", ""))
        packaged["evidence_heading_path"] = evidence.get("heading_path", chunk.get("heading_path", ""))
        packaged["evidence_page_start"] = evidence.get("page_start", chunk.get("page_start"))
        packaged["evidence_page_end"] = evidence.get("page_end", chunk.get("page_end"))
        packaged["evidence_card"] = {
            "chunk_id": evidence["chunk_id"],
            "matched_chunk_id": chunk["chunk_id"],
            "matched_level": chunk.get("level"),
            "evidence_level": evidence.get("level"),
            "doc_id": evidence.get("doc_id", chunk.get("doc_id")),
            "doc_title": evidence.get("doc_title", chunk.get("doc_title")),
            "collection": evidence.get("collection", chunk.get("collection")),
            "trust_tier": evidence.get("trust_tier", chunk.get("trust_tier")),
            "section_title": evidence.get("section_title", chunk.get("section_title")),
            "chunk_type": chunk.get("chunk_type"),
            "anatomy_tags": chunk.get("anatomy_tags", []),
            "risk_tags": chunk.get("risk_tags", []),
            "phase_scope": chunk.get("phase_scope", []),
            "score": round(float(score), 4),
            "page_start": evidence.get("page_start", chunk.get("page_start")),
            "page_end": evidence.get("page_end", chunk.get("page_end")),
        }
        return packaged

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        alpha: Optional[float] = None,
        filter_collections: Optional[list[str]] = None,
        filter_chunk_types: Optional[list[str]] = None,
        boost_collections: bool = True,
        question_type: Optional[str] = None,
        visual_terms: Optional[list[str]] = None,
        classes_detected: Optional[dict] = None,
        extra_terms: Optional[list[str]] = None,
        expand_parents: bool = True,
        return_evidence_cards: bool = False,
    ):
        alpha = alpha if alpha is not None else self.alpha
        pool = max(top_k * 8, 40, self.rerank_top_n if self.use_reranker else 0)
        query_bundle = self.build_query_bundle(
            query,
            question_type=question_type,
            extra_terms=extra_terms,
            visual_terms=visual_terms,
            classes_detected=classes_detected,
        )

        bm25_results = self.retrieve_bm25(query_bundle, top_k=pool)

        if self.retrieval_mode == "bm25_only" or self.encoder is None or self.faiss_index is None:
            ranked = self._apply_priors(
                bm25_results,
                query_bundle,
                filter_collections=filter_collections,
                filter_chunk_types=filter_chunk_types,
                boost_collections=boost_collections,
            )
        else:
            dense_results = self.retrieve_dense(query_bundle, top_k=pool)
            fused = self._fuse_rrf(bm25_results, dense_results, k=60)

            if alpha is not None and abs(alpha - 0.5) > 1e-6:
                dense_ids = {c["chunk_id"] for c, _ in dense_results[: max(top_k * 4, 20)]}
                bm25_ids = {c["chunk_id"] for c, _ in bm25_results[: max(top_k * 4, 20)]}
                adjusted = []
                for chunk, score in fused:
                    mix = 1.0
                    if chunk["chunk_id"] in dense_ids:
                        mix += alpha * 0.15
                    if chunk["chunk_id"] in bm25_ids:
                        mix += (1 - alpha) * 0.15
                    adjusted.append((chunk, score * mix))
                fused = adjusted

            ranked = self._apply_priors(
                fused,
                query_bundle,
                filter_collections=filter_collections,
                filter_chunk_types=filter_chunk_types,
                boost_collections=boost_collections,
            )

        selected = self._adaptive_select(ranked, top_k=top_k)
        packaged = [
            (self._package_candidate(chunk, score, expand_parents=expand_parents), float(score))
            for chunk, score in selected
        ]

        rerank_query = query_bundle["dense_query"]
        packaged = self.rerank_candidates(rerank_query, packaged, top_k)
        if return_evidence_cards:
            return [chunk["evidence_card"] for chunk, _ in packaged]
        return packaged[:top_k]

    def retrieve_hybrid(self, query, top_k=RETRIEVAL_TOP_K, **kw):
        return self.retrieve(query, top_k=top_k, **kw)

    def evaluate_retrieval(self, eval_path: str, top_k: int = RETRIEVAL_TOP_K):
        with open(eval_path, encoding="utf-8") as f:
            eval_data = json.load(f)
        question_meta = {}
        if QUESTIONS_FILE.exists():
            try:
                with open(QUESTIONS_FILE, encoding="utf-8") as f:
                    question_rows = json.load(f)
                question_meta = {row["qid"]: row for row in question_rows}
            except Exception:
                question_meta = {}

        hits = 0
        total = len(eval_data)
        report = []

        for item in eval_data:
            qid = item["qid"]
            query = item["question"]
            acceptable_needles = item.get("acceptable_needles") or [item.get("min_acceptable_chunk_contains", "")]
            acceptable_needles = [
                str(needle).strip().lower()
                for needle in acceptable_needles
                if str(needle).strip()
            ]
            question_type = item.get("question_type")
            qmeta = question_meta.get(qid, {})

            results = self.retrieve(
                query,
                top_k=top_k,
                question_type=question_type,
                classes_detected=qmeta.get("classes_detected"),
            )
            found = any(
                any(
                    needle in (c.get("evidence_raw_text") or c.get("text", "")).lower()
                    or needle in (c.get("evidence_text") or "").lower()
                    for needle in acceptable_needles
                )
                for c, _ in results
            )

            status = "HIT" if found else "MISS"
            hits += int(found)
            needles_label = ", ".join(acceptable_needles[:4])
            if len(acceptable_needles) > 4:
                needles_label += ", ..."

            if results:
                top = results[0][0]
                line = (
                    f"  {qid} [{status}] needles='{needles_label}' "
                    f"top1: [{top.get('collection','?')}/{top.get('chunk_type','?')}] "
                    f"{(top.get('evidence_raw_text') or top.get('text',''))[:80]}..."
                )
            else:
                line = f"  {qid} [{status}] needles='{needles_label}' (no results)"
            report.append(line)

        recall = hits / total if total else 0
        print(f"\n{'='*70}")
        print(_safe_console_text(f"Retrieval Eval - Recall@{top_k}: {hits}/{total} = {recall:.0%}"))
        print(f"{'='*70}")
        for line in report:
            print(_safe_console_text(line))

        return {"recall": recall, "hits": hits, "total": total}


SurgicalRetriever = SurgicalRetrieverV2


def main():
    retriever = SurgicalRetrieverV2()

    test_queries = [
        ("What must be confirmed before clipping the cystic duct?", "safety_verification"),
        ("critical view of safety requirements", "safety_verification"),
        ("bile duct injury prevention", "risk_pitfall"),
        ("when to defer if anatomy unclear", "risk_pitfall"),
        ("instruments used in laparoscopic cholecystectomy", "recognition"),
        ("gallbladder fundus identification", "anatomy_landmark"),
        ("bleeding during dissection hemostasis", "risk_pitfall"),
        ("liver bed dissection gallbladder separation", "workflow_phase"),
    ]

    for q, qtype in test_queries:
        print(f"\n{'-'*60}")
        print(f"Query: {q}  |  qtype={qtype}")
        results = retriever.retrieve(q, top_k=3, question_type=qtype)
        for rank, (chunk, score) in enumerate(results, 1):
            coll = chunk.get("collection", "?")
            ctype = chunk.get("chunk_type", "?")
            print(f"  #{rank} [{score:.4f}] [{coll}/{ctype}] {chunk['doc_title']}")
            print(f"       matched={chunk['matched_chunk_id']} evidence={chunk['evidence_chunk_id']}")
            print(f"       {(chunk.get('evidence_raw_text') or chunk['text'])[:150]}...")

    eval_path = RETRIEVAL_EVAL_FILE
    if eval_path.exists():
        retriever.evaluate_retrieval(str(eval_path))
    else:
        print(f"\n[!] Eval file not found: {eval_path}")


if __name__ == "__main__":
    main()
