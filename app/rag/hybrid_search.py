# Hybrid search - combines semantic similarity with simple keyword matching
import os
from app.rag.retriever import load_store


def keyword_score(query, text):
    # Very simple keyword overlap score
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not query_words:
        return 0.0
    overlap = query_words & text_words
    return len(overlap) / len(query_words)


def hybrid_search(store_path, query, k=4, alpha=0.6):
    # alpha controls the weight of semantic vs keyword
    # alpha=1.0 -> only semantic, alpha=0.0 -> only keyword
    if not os.path.exists(store_path):
        return []

    vs = load_store(store_path)
    # Pull more candidates and rerank
    semantic_results = vs.similarity_search_with_score(query, k=k * 3)

    scored = []
    for doc, sem_score in semantic_results:
        # FAISS returns distance (lower = better) -> convert to similarity (higher = better)
        sem_similarity = 1.0 / (1.0 + sem_score)
        kw_similarity = keyword_score(query, doc.page_content)
        combined = alpha * sem_similarity + (1 - alpha) * kw_similarity
        scored.append((doc, combined))

    # Sort by combined score
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:k]]


def hybrid_search_all(query, k=3, alpha=0.6):
    # Run hybrid search across all three vectorstores and combine
    results = []
    for store_path in [
        "vectorstores/finance_10k",
        "vectorstores/financial_news",
        "vectorstores/market_analysis",
    ]:
        results.extend(hybrid_search(store_path, query, k=k, alpha=alpha))
    return results
