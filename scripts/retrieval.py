"""
retrieval_v2.py — Hybrid retrieval with collection-aware boosting.

Usage:
    from retrieval_v2 import SurgicalRetrieverV2
    retriever = SurgicalRetrieverV2("docs/chunks/chunks_v2.jsonl")
    results = retriever.retrieve("critical view of safety")
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CHUNKS_FILE,
    DENSE_MODEL_NAME,
    HYBRID_ALPHA,
    RETRIEVAL_TOP_K,
    RETRIEVAL_MODE,
    HF_TOKEN,
    HF_CACHE_DIR,
    HF_LOCAL_FILES_ONLY,
    COLLECTION_PRIORITY,
    RETRIEVAL_EVAL_FILE,
)


class SurgicalRetrieverV2:

    def __init__(
        self,
        chunks_path: Optional[str] = None,
        dense_model: str = DENSE_MODEL_NAME,
        alpha: float = HYBRID_ALPHA,
    ):
        chunks_path = chunks_path or str(CHUNKS_FILE)

        # Load chunks (encoding-safe)
        self.chunks: list[dict] = []
        with open(chunks_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        self.texts = [c["text"] for c in self.chunks]
        self.alpha = alpha
        self.retrieval_mode = RETRIEVAL_MODE
        self.encoder = None
        self.faiss_index = None
        print(f"[RetrieverV2] Loaded {len(self.chunks)} chunks from {chunks_path}")

        if not self.chunks:
            raise ValueError("Chunk file is empty. Run build_corpus_v2.py first.")

        # Log collection distribution
        collections = {}
        for c in self.chunks:
            coll = c.get("collection", "unknown")
            collections[coll] = collections.get(coll, 0) + 1
        for coll, count in sorted(collections.items()):
            print(f"  {coll}: {count} chunks")

        # BM25
        from rank_bm25 import BM25Okapi
        tokenised = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenised)
        print("[RetrieverV2] BM25 index built")

        if self.retrieval_mode == "bm25_only":
            print("[RetrieverV2] Dense retrieval disabled by RETRIEVAL_MODE=bm25_only")
            return

        # Dense
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            print(f"[RetrieverV2] Loading dense model: {dense_model} ...")
            t0 = time.time()
            self.encoder = SentenceTransformer(
                dense_model,
                cache_folder=HF_CACHE_DIR or None,
                local_files_only=HF_LOCAL_FILES_ONLY,
                token=HF_TOKEN or None,
            )
            embeddings = self.encoder.encode(
                self.texts, show_progress_bar=True,
                batch_size=64, normalize_embeddings=True,
            )
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(embeddings.astype("float32"))
            print(f"[RetrieverV2] Dense index built ({dim}d, {time.time()-t0:.1f}s)")
        except Exception as e:
            self.retrieval_mode = "bm25_only"
            self.encoder = None
            self.faiss_index = None
            print(f"[RetrieverV2] Dense retrieval unavailable, falling back to BM25-only: {e}")
            print(
                "[RetrieverV2] To enable hybrid retrieval, pre-download the embedding model "
                "or set HF_CACHE_DIR / HF_LOCAL_FILES_ONLY appropriately."
            )

    # ─── Sparse ──────────────────────────────────────────────────

    def retrieve_bm25(self, query: str, top_k: int = RETRIEVAL_TOP_K):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    # ─── Dense ───────────────────────────────────────────────────

    def retrieve_dense(self, query: str, top_k: int = RETRIEVAL_TOP_K):
        if self.encoder is None or self.faiss_index is None:
            return []
        q_emb = self.encoder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.faiss_index.search(q_emb, top_k)
        return [
            (self.chunks[indices[0][i]], float(scores[0][i]))
            for i in range(top_k)
            if indices[0][i] < len(self.chunks)
        ]

    # ─── Hybrid with collection boosting ─────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        alpha: Optional[float] = None,
        filter_collections: Optional[list[str]] = None,
        filter_chunk_types: Optional[list[str]] = None,
        boost_collections: bool = True,
    ):
        alpha = alpha if alpha is not None else self.alpha
        pool = top_k * 4

        bm25_results = self.retrieve_bm25(query, top_k=pool)

        if self.retrieval_mode == "bm25_only" or self.encoder is None or self.faiss_index is None:
            ranked = []
            for chunk, score in bm25_results:
                if filter_collections and chunk.get("collection") not in filter_collections:
                    continue
                if filter_chunk_types and chunk.get("chunk_type") not in filter_chunk_types:
                    continue
                final_score = score
                if boost_collections:
                    coll = chunk.get("collection", "general")
                    final_score *= COLLECTION_PRIORITY.get(coll, 0.7)
                ranked.append((chunk, float(final_score)))
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

        dense_results = self.retrieve_dense(query, top_k=pool)

        def _norm(results):
            scores = [s for _, s in results]
            mn, mx = min(scores, default=0), max(scores, default=1)
            rng = (mx - mn) or 1.0
            return [(c, (s - mn) / rng) for c, s in results]

        bm25_n  = _norm(bm25_results)
        dense_n = _norm(dense_results)

        combined: dict[str, dict] = {}

        for chunk, score in bm25_n:
            cid = chunk["chunk_id"]
            combined.setdefault(cid, {"chunk": chunk, "score": 0.0})
            combined[cid]["score"] += (1 - alpha) * score

        for chunk, score in dense_n:
            cid = chunk["chunk_id"]
            combined.setdefault(cid, {"chunk": chunk, "score": 0.0})
            combined[cid]["score"] += alpha * score

        # Collection boosting
        if boost_collections:
            for cid, entry in combined.items():
                coll = entry["chunk"].get("collection", "general")
                boost = COLLECTION_PRIORITY.get(coll, 0.7)
                entry["score"] *= boost

        # Filters
        if filter_collections:
            combined = {
                cid: e for cid, e in combined.items()
                if e["chunk"].get("collection") in filter_collections
            }
        if filter_chunk_types:
            combined = {
                cid: e for cid, e in combined.items()
                if e["chunk"].get("chunk_type") in filter_chunk_types
            }

        ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return [(r["chunk"], r["score"]) for r in ranked[:top_k]]

    # ─── Backward compat ─────────────────────────────────────────

    def retrieve_hybrid(self, query, top_k=RETRIEVAL_TOP_K, **kw):
        return self.retrieve(query, top_k=top_k, **kw)

    # ─── Evaluation ──────────────────────────────────────────────

    def evaluate_retrieval(self, eval_path: str, top_k: int = RETRIEVAL_TOP_K):
        with open(eval_path, encoding="utf-8") as f:
            eval_data = json.load(f)

        hits = 0
        total = len(eval_data)
        report = []

        for item in eval_data:
            qid = item["qid"]
            query = item["question"]
            needle = item["min_acceptable_chunk_contains"].lower()

            results = self.retrieve(query, top_k=top_k)
            found = any(needle in c["text"].lower() for c, _ in results)

            status = "HIT" if found else "MISS"
            hits += int(found)

            if results:
                top = results[0][0]
                line = (f"  {qid} [{status}]  needle='{needle}'  "
                        f"top1: [{top.get('collection','?')}/{top.get('chunk_type','?')}] "
                        f"{top['text'][:80]}...")
            else:
                line = f"  {qid} [{status}]  needle='{needle}'  (no results)"
            report.append(line)

        recall = hits / total if total else 0
        print(f"\n{'='*70}")
        print(f"Retrieval Eval — Recall@{top_k}: {hits}/{total} = {recall:.0%}")
        print(f"{'='*70}")
        for line in report:
            print(line)

        return {"recall": recall, "hits": hits, "total": total}


SurgicalRetriever = SurgicalRetrieverV2


# ─── Standalone ──────────────────────────────────────────────────────

def main():
    retriever = SurgicalRetrieverV2()

    test_queries = [
        "What must be confirmed before clipping the cystic duct?",
        "critical view of safety requirements",
        "bile duct injury prevention",
        "when to defer if anatomy unclear",
        "instruments used in laparoscopic cholecystectomy",
        "gallbladder fundus identification",
        "bleeding during dissection hemostasis",
        "liver bed dissection gallbladder separation",
    ]

    for q in test_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {q}")
        results = retriever.retrieve(q, top_k=3)
        for rank, (chunk, score) in enumerate(results, 1):
            coll = chunk.get("collection", "?")
            ctype = chunk.get("chunk_type", "?")
            print(f"  #{rank} [{score:.3f}] [{coll}/{ctype}] {chunk['doc_title']}")
            print(f"         {chunk['text'][:150]}...")

    # Eval
    eval_path = RETRIEVAL_EVAL_FILE
    if eval_path.exists():
        retriever.evaluate_retrieval(str(eval_path))
    else:
        print(f"\n[!] Eval file not found: {eval_path}")


if __name__ == "__main__":
    main()
