# src/features/heuristic_extractor.py
"""
Heuristic configurations and taxomomies cloned from recruitgpt-x.
"""

# Consulting firms
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "mindtree", "ltimindtree", "hcl", 
    "tech mahindra", "deloitte consulting", "ibm consulting"
}

# Core skill phrases from recruitgpt-x
CORE_SKILL_PHRASES = {
    "embeddings", "embedding", "information retrieval", "retrieval", 
    "ranking systems", "learning to rank", "vector search", "vector database", 
    "vector representations", "semantic search", "hybrid search", "hybrid retrieval", 
    "faiss", "qdrant", "pinecone", "weaviate", "milvus", "opensearch", 
    "elasticsearch", "pgvector", "sentence transformers", "sentence transformer", 
    "ndcg", "mrr"
}

# Secondary skill phrases
SECONDARY_SKILL_PHRASES = {
    "nlp", "llm", "fine-tuning", "fine tuning", "lora", "qlora", "mlops", 
    "mlflow", "hugging face", "transformers", "deep learning", "machine learning", 
    "recommendation", "recommendation systems"
}

# TIER_1_SKILLS
TIER_1_SKILLS = CORE_SKILL_PHRASES.union(SECONDARY_SKILL_PHRASES)

# Build a dictionary for JD_TAXONOMY that contains all core and secondary skill phrases
# with arbitrary weights (e.g. 1.0) so that audit_submission.py anti-hallucination check
# runs over all these terms.
JD_TAXONOMY = {}
for skill in CORE_SKILL_PHRASES:
    JD_TAXONOMY[skill] = 1.0
for skill in SECONDARY_SKILL_PHRASES:
    JD_TAXONOMY[skill] = 0.5