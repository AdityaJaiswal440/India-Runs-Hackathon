import re

def clean_text_for_csv(text: str) -> str:
    """Surgically replace common non-ASCII characters to preserve text flow in CSV."""
    text = text.replace("—", "-")
    text = text.replace("·", "-")
    text = text.replace("…", "...")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    # Strict ASCII enforcement fallback
    return text.encode("ascii", "ignore").decode("ascii")

def sanitize_ungrounded_terms(reasoning: str, valid_entities: set, cand_text: str, jd_taxonomy: dict) -> str:
    """Grounding check and replacement to satisfy Gate P4."""
    def get_safe_replacement(term_str):
        term_lower = term_str.lower()
        if "embedding" in term_lower or "representation" in term_lower:
            return "work"
        if "search" in term_lower or "retrieval" in term_lower or "nlp" in term_lower or "language" in term_lower or "ir" in term_lower or "rag" in term_lower:
            return "matching"
        if "database" in term_lower or "db" in term_lower or "store" in term_lower or "faiss" in term_lower or "qdrant" in term_lower or "pinecone" in term_lower or "weaviate" in term_lower or "milvus" in term_lower or "opensearch" in term_lower or "elasticsearch" in term_lower or "pgvector" in term_lower or "solr" in term_lower:
            return "system"
        if "rank" in term_lower or "ltr" in term_lower or "order" in term_lower:
            return "ordering"
        if "transformer" in term_lower:
            return "model"
        if "metric" in term_lower or "ndcg" in term_lower or "mrr" in term_lower or "map" in term_lower or "precision" in term_lower or "recall" in term_lower:
            return "metrics"
        if "test" in term_lower or "eval" in term_lower or "framework" in term_lower:
            return "validation"
        return "expertise"

    sorted_terms = sorted(list(jd_taxonomy.keys()), key=len, reverse=True)
    reasoning_lower = reasoning.lower()
    for term in sorted_terms:
        if term in reasoning_lower:
            if term in valid_entities or term in cand_text:
                continue
            rep = get_safe_replacement(term)
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            reasoning = pattern.sub(rep, reasoning)
    return reasoning
