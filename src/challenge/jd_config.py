"""Job description constants for the Senior AI Engineer ranking role.

All phrase lists, title tiers, skill taxonomies, and experience thresholds
derived from the target JD live here. Keeping them in one place means a
role change only requires editing this file — no scoring logic changes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Title tiers — ordered by alignment with the JD mandate
# ---------------------------------------------------------------------------

STRONG_TITLES = (
    "senior ai engineer",
    "ai engineer",
    "machine learning engineer",
    "ml engineer",
    "senior machine learning engineer",
    "staff ml engineer",
    "applied scientist",
    "senior data scientist",
    "recommendation systems engineer",
    "applied ml engineer",
    "senior nlp engineer",
    "search engineer",
)

GOOD_TITLES = (
    "data scientist",
    "ml scientist",
    "nlp engineer",
    "recommendation engineer",
    "software engineer",
    "backend engineer",
)

WEAK_TITLES = (
    "hr manager",
    "accountant",
    "sales executive",
    "marketing manager",
    "graphic designer",
    "content writer",
    "customer support",
    "civil engineer",
    "mechanical engineer",
    "operations manager",
    "project manager",
    "business analyst",
    "qa engineer",
)

# Research-ladder titles without applied/product delivery context
RESEARCH_ONLY_TITLES = (
    "research scientist",
    "principal scientist",
)

# ---------------------------------------------------------------------------
# Skill taxonomies
# ---------------------------------------------------------------------------

# Core IR/ranking skills — primary signal for JD fit
CORE_SKILL_PHRASES = (
    "embeddings",
    "embedding",
    "information retrieval",
    "retrieval",
    "ranking systems",
    "learning to rank",
    "vector search",
    "vector database",
    "vector representations",
    "semantic search",
    "hybrid search",
    "hybrid retrieval",
    "faiss",
    "qdrant",
    "pinecone",
    "weaviate",
    "milvus",
    "opensearch",
    "elasticsearch",
    "pgvector",
    "sentence transformers",
    "sentence transformer",
    "ndcg",
    "mrr",
    "rag",
    "retrieval augmented generation",
)

# Broader ML tooling — secondary signal, not counted as IR depth
SECONDARY_SKILL_PHRASES = (
    "nlp",
    "llm",
    "large language model",
    "genai",
    "generative ai",
    "fine-tuning",
    "fine tuning",
    "lora",
    "qlora",
    "mlops",
    "mlflow",
    "hugging face",
    "transformers",
    "deep learning",
    "machine learning",
    "recommendation",
    "recommendation systems",
)

# General infrastructure/language skills — not counted as IR depth signals
GENERAL_ML_SKILLS = (
    "python",
    "pytorch",
    "tensorflow",
    "tf",
    "k8s",
    "kubernetes",
    "gcp",
    "google cloud",
)

# Skills that inflate profile noise without adding IR signal
HONEYPOT_SKILL_NOISE = (
    "html",
    "tailwind",
    "photoshop",
    "css",
    "react",
    "javascript",
    "excel",
    "powerpoint",
    "word",
)

# Domains outside the JD mandate — presence penalises fit without IR depth
CV_SPEECH_ROBOTICS = (
    "computer vision",
    "object detection",
    "speech recognition",
    "asr",
    "robotics",
    "autonomous driving",
)

# Orchestration-layer frameworks — noise when IR depth is absent
FRAMEWORK_NOISE = (
    "langchain",
    "llamaindex",
    "autogen",
    "crewai",
)

# ---------------------------------------------------------------------------
# Company signals
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = (
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
    "ltimindtree",
    "hcl",
    "tech mahindra",
    "deloitte consulting",
    "ibm consulting",
)

# Indian product startups — JD values shipper mindset over FAANG ladder
STARTUP_BOOST_SIGNALS = {
    "zomato",
    "phonepe",
    "swiggy",
    "flipkart",
    "cred",
    "razorpay",
    "krutrim",
    "sarvam",
    "rephrase",
    "unacademy",
    "nykaa",
    "zoho",
    "aganitha",
    "niramai",
    "verloop",
}

# Current FAANG role is a yellow flag for founding-team fit, not a green one
FAANG_CURRENT_PENALTY = {
    "google",
    "meta",
    "facebook",
    "apple",
    "microsoft",
    "netflix",
}

# ---------------------------------------------------------------------------
# Location preferences
# ---------------------------------------------------------------------------

PREFERRED_LOCATIONS = ("pune", "noida")

INDIA_LOCATIONS = (
    "pune",
    "noida",
    "bangalore",
    "bengaluru",
    "hyderabad",
    "mumbai",
    "delhi",
    "gurgaon",
    "gurugram",
    "chennai",
    "india",
)

# ---------------------------------------------------------------------------
# Experience thresholds (years)
# ---------------------------------------------------------------------------

EXP_MIN = 4.0
EXP_IDEAL_LO = 5.0
EXP_IDEAL_HI = 9.0
EXP_MAX = 15.0

# ---------------------------------------------------------------------------
# JD document and career matching phrases
# ---------------------------------------------------------------------------

JOB_TITLE = "Senior AI Engineer — Founding Team"
JOB_COMPANY = "Redrob AI"

# Compact JD excerpt used for offline TF-IDF career similarity
JD_DOCUMENT = """
senior ai engineer founding team embeddings retrieval ranking hybrid search vector database
production deployed users evaluation ndcg mrr map learning to rank recommendation systems
sentence transformers faiss pinecone qdrant weaviate milvus opensearch elasticsearch python
fine-tuning lora shipper product company recruiter matching marketplace hiring
"""

# Weighted phrases for career-description matching
CAREER_JD_WEIGHTS = {
    "recommendation system": 0.22,
    "recommendation systems": 0.22,
    "ranking system": 0.20,
    "search system": 0.18,
    "vector search": 0.18,
    "hybrid search": 0.16,
    "hybrid retrieval": 0.16,
    "embedding": 0.14,
    "retrieval": 0.14,
    "ndcg": 0.12,
    "learning to rank": 0.12,
    "deployed to": 0.10,
    "production": 0.08,
    "a/b test": 0.08,
    "recruiter": 0.06,
}

JD_OVERLAP_PHRASES = tuple(CAREER_JD_WEIGHTS.keys()) + (
    "embeddings",
    "ranking",
    "vector",
    "serving",
    "pipeline",
    "matching",
)

PRODUCTION_SIGNAL_PHRASES = (
    "production",
    "deployed",
    "serving",
    "shipped",
    "users",
    "a/b",
    "ndcg",
    "mrr",
    "retrieval",
    "embedding",
    "vector",
    "ranking",
    "pipeline",
    "scale",
    "online",
    "offline",
)
