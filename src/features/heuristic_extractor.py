# src/features/heuristic_extractor.py
import datetime
import re
from src.features.honeypot import TIER_1_SKILLS, safe_date

# Blacklist of consulting firms for company type score
CONSULTING_FIRMS = {
    "tata consultancy services", "tcs", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "hcl technologies", "tech mahindra", 
    "hexaware", "mphasis", "ltimindtree", "l&t infotech"
}

# Industry relevance terms
INDUSTRY_RELEVANT_TERMS = {
    "ai", "ml", "machine learning", "artificial intelligence", "search", 
    "information retrieval", "recruiting", "hrtech", "hr tech", "hr-tech", 
    "software", "technology", "internet", "saas", "recruitment", 
    "human resources", "staffing"
}

# Skill sets for CV Domain Mismatch Penalty
CV_SPEECH_ROBOTICS_SKILLS = {
    "computer vision", "cv", "robotics", "speech recognition", "speech", 
    "cv engineer", "image processing", "object detection", "image segmentation", 
    "autonomous vehicles", "speech-to-text", "text-to-speech", "ros", "opencv", 
    "pytorch lightning", "yolo", "cnn", "image classification", "point cloud", 
    "slam", "lidar", "radar", "speech classification", "asr", "tts", "whisper"
}

NLP_IR_EVIDENCE_SKILLS = {
    "nlp", "natural language processing", "information retrieval", "ir", "search", 
    "retrieval", "embeddings", "embedding", "vector search", "hybrid search", "bm25", 
    "rag", "sentence-transformers", "bert", "llm", "large language models", 
    "transformers", "text mining", "word2vec", "spacy", "nltk", "lucene", "solr", 
    "elasticsearch", "opensearch", "milvus", "qdrant", "pinecone", "weaviate", "faiss"
}

# Skill taxonomy definition
TIER_1_TAXONOMY = {
    "embeddings": 1.0, "embedding": 1.0, "sentence-transformers": 1.0, "sentence transformers": 1.0,
    "sbert": 0.95, "openai embeddings": 0.95, "text-embedding-ada-002": 0.9, "bge": 0.95,
    "bge-large": 0.95, "bge-m3": 0.95, "e5": 0.95, "multilingual-e5": 0.9, "embedding drift": 0.85,
    "index refresh": 0.8, "reindexing": 0.75,
    "vector database": 1.0, "vector db": 1.0, "vector store": 0.95, "pinecone": 1.0,
    "weaviate": 1.0, "qdrant": 1.0, "milvus": 1.0, "opensearch": 0.9, "elasticsearch": 0.9,
    "elastic search": 0.9, "solr": 0.85, "faiss": 1.0, "ann": 0.8, "approximate nearest neighbor": 0.8,
    "hybrid search": 0.95, "hybrid retrieval": 0.95, "retrieval": 1.0, "rag": 0.9,
    "retrieval augmented generation": 0.9,
    "python": 0.6,
    "ndcg": 0.9, "mrr": 0.9, "map": 0.7, "precision@k": 0.85, "recall@k": 0.85,
    "evaluation framework": 0.85, "ab testing": 0.8, "a/b testing": 0.8, "learning to rank": 0.9,
    "ltr": 0.9, "offline-to-online correlation": 0.75
}

TIER_2_TAXONOMY = {
    "lora": 0.6, "qlora": 0.6, "peft": 0.6, "fine-tuning": 0.65, "fine tuning": 0.65, "finetuning": 0.65,
    "xgboost": 0.55, "lightgbm": 0.55, "neural learning-to-rank": 0.55, "distributed systems": 0.5,
    "inference optimization": 0.5, "large-scale inference": 0.5,
    "open source": 0.4, "open-source contributions": 0.4,
    "hr tech": 0.5, "hr-tech": 0.5, "recruiting": 0.4, "hr": 0.5, "talent": 0.4, "marketplace": 0.5
}

JD_TAXONOMY = TIER_2_TAXONOMY.copy()
JD_TAXONOMY.update(TIER_1_TAXONOMY)

MAX_POSSIBLE_SKILL_SUM = sum(JD_TAXONOMY.values())
if MAX_POSSIBLE_SKILL_SUM == 0:
    MAX_POSSIBLE_SKILL_SUM = 1.0

# Company size mapping for Startup Trajectory
SIZE_TIER_ORDER = {
    "1-10": 0,
    "11-50": 1,
    "51-200": 2,
    "201-500": 3,
    "501-1000": 4,
    "1001-5000": 5,
    "5001-10000": 6,
    "10001+": 7
}

# Core clusters for Skill Gap Coverage
embeddings_cluster = {"embeddings", "embedding", "sentence-transformers", "sentence transformers", "bge", "e5"}
retrieval_cluster = {"retrieval", "rag", "hybrid search", "learning to rank", "ltr"}
vector_db_cluster = {"vector database", "vector db", "pinecone", "milvus", "qdrant", "weaviate", "faiss", "opensearch", "elasticsearch"}

def compute_skill_match_score(candidate: dict) -> float:
    """
    Computes trust-weighted skill match score (anti-stuffing) per Module 5.2,
    and applies the LangChain-only penalty per Module 5.9.1.
    """
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    assessments = signals.get("skill_assessment_scores", {}) or {}
    
    contribution_sum = 0.0
    for s in skills:
        if not s:
            continue
        sname = (s.get("name") or "").lower().strip()
        if sname not in JD_TAXONOMY:
            continue
            
        base_weight = JD_TAXONOMY[sname]
        
        # 1. Proficiency factor
        prof = (s.get("proficiency") or "").lower().strip()
        if prof == "expert":
            prof_factor = 1.00
        elif prof == "advanced":
            prof_factor = 0.85
        elif prof == "intermediate":
            prof_factor = 0.60
        else:
            prof_factor = 0.30  # beginner/missing
            
        # 2. Duration Gate
        duration = s.get("duration_months")
        if duration is None or int(duration) == 0:
            continue
        duration_val = int(duration)
        
        # 3. Duration Trust
        duration_trust = min(duration_val / 24.0, 1.0)
        
        # 4. Assessment Factor
        assess_score = assessments.get(sname) or assessments.get(s.get("name"))
        if assess_score is not None:
            assessment_factor = float(assess_score) / 100.0
        else:
            assessment_factor = 0.65
            
        # 5. Endorsement Factor
        endorsements = s.get("endorsements", 0) or 0
        endorsement_factor = min(int(endorsements) / 25.0, 1.0)
        
        contribution = base_weight * (
            0.25 * prof_factor +
            0.40 * duration_trust +
            0.25 * assessment_factor +
            0.10 * endorsement_factor
        )
        contribution_sum += contribution
        
    skill_match_score = min(contribution_sum / MAX_POSSIBLE_SKILL_SUM, 1.0)
    
    # Apply LangChain-only penalty
    has_langchain = any((s.get("name") or "").lower().strip() == "langchain" for s in skills if s)
    RETRIEVAL_EVIDENCE_SKILLS = {
        "rag", "retrieval augmented generation", "pinecone", "faiss", "milvus", 
        "qdrant", "weaviate", "embeddings", "embedding", "vector database", 
        "vector db", "vector store", "hybrid search", "hybrid retrieval"
    }
    has_retrieval_evidence = any((s.get("name") or "").lower().strip() in RETRIEVAL_EVIDENCE_SKILLS for s in skills if s)
    
    career_history = candidate.get("career_history", []) or []
    earliest_year = None
    for job in career_history:
        if job:
            start = safe_date(job.get("start_date"))
            if start:
                if earliest_year is None or start.year < earliest_year:
                    earliest_year = start.year
                    
    # LangChain penalty: langchain skill, no retrieval evidence, career began 2023 or later
    if has_langchain and not has_retrieval_evidence and earliest_year is not None and earliest_year >= 2023:
        skill_match_score *= 0.5
        
    return float(max(min(skill_match_score, 1.0), 0.0))

def compute_skill_gap_coverage(candidate: dict) -> float:
    """
    Computes Skill Gap Coverage per Module 5.7.
    """
    skills = candidate.get("skills", []) or []
    trusted_skills = set()
    for s in skills:
        if s:
            name = (s.get("name") or "").lower().strip()
            duration = s.get("duration_months", 0) or 0
            if duration >= 6:
                trusted_skills.add(name)
                
    covered_clusters = 0
    if trusted_skills.intersection(embeddings_cluster):
        covered_clusters += 1
    if trusted_skills.intersection(retrieval_cluster):
        covered_clusters += 1
    if trusted_skills.intersection(vector_db_cluster):
        covered_clusters += 1
        
    return float(covered_clusters / 3.0)

def compute_career_score(candidate: dict) -> float:
    """
    Computes Career Quality Composite per Module 5.8,
    applies Industry Relevance Multiplier per Module 5.8.5,
    and CV/Speech/Robotics Domain-Mismatch Trap per Module 5.9.2.
    """
    career_history = candidate.get("career_history", []) or []
    profile = candidate.get("profile", {}) or {}
    
    # -------------------------------------------------------------
    # 5.8.1 Company Type Score (Consulting Penalty)
    # -------------------------------------------------------------
    consulting_months = 0
    product_months = 0
    
    for job in career_history:
        if not job:
            continue
        comp = (job.get("company") or "").lower()
        duration = job.get("duration_months", 0) or 0
        
        # Word boundary regex matching consulting firms
        is_consulting = False
        for firm in CONSULTING_FIRMS:
            pattern = rf"\b{re.escape(firm)}\b"
            if re.search(pattern, comp):
                is_consulting = True
                break
                
        if is_consulting:
            consulting_months += duration
        else:
            product_months += duration
            
    company_type_score = max(product_months / (consulting_months + product_months + 1.0), 0.02)
    
    # -------------------------------------------------------------
    # 5.8.2 Job Stability Score
    # -------------------------------------------------------------
    if len(career_history) < 2:
        stability_score = 1.0
    else:
        short_stints = sum(1 for job in career_history if job and (job.get("duration_months", 0) or 0) < 18 and job.get("is_current") is not True)
        stability_score = max(1.0 - (short_stints / len(career_history) * 0.75), 0.1)
        
    # -------------------------------------------------------------
    # 5.8.3 Title Relevance Score
    # -------------------------------------------------------------
    strong_titles = {"ml engineer", "machine learning", "ai engineer", "applied scientist", "nlp engineer", "search engineer", "ranking", "recommendation", "retrieval"}
    relevant_titles = strong_titles.union({"data scientist", "research engineer", "ranking engineer", "backend engineer", "software engineer"})
    
    tech_architect_qualifiers = {"ml", "ai", "data", "search", "nlp", "cloud", "platform"}
    neutral_architect_terms = {"software", "technical", "solution", "enterprise", "principal"}
    
    disqualified_titles = {"marketing manager", "hr manager", "content writer", "sales", "business analyst", "scrum master", "project manager", "recruiter", "computer vision", "robotics", "speech recognition", "cv engineer"}
    
    weighted_months = 0.0
    total_months = 0
    
    for job in career_history:
        if not job:
            continue
        title = (job.get("title") or "").lower()
        duration = job.get("duration_months", 0) or 0
        total_months += duration
        
        weight = 0.0
        # Precedence check
        if "architect" in title:
            # Tech architect check
            if any(qual in title for qual in tech_architect_qualifiers):
                # Tech architect falls through to standard matching
                if any(t in title for t in strong_titles):
                    weight = 1.0
                elif any(t in title for t in relevant_titles):
                    weight = 0.5
            elif any(term in title for term in neutral_architect_terms):
                weight = 0.25  # Neutral Architect
            else:
                weight = 0.0   # Disqualified Architect
        elif any(t in title for t in disqualified_titles):
            weight = 0.0
        elif any(t in title for t in strong_titles):
            weight = 1.0
        elif any(t in title for t in relevant_titles):
            weight = 0.5
        else:
            weight = 0.0 # unmatched
            
        weighted_months += weight * duration
        
    title_relevance_score = weighted_months / max(total_months, 1)
    
    # -------------------------------------------------------------
    # 5.8.4 Startup Trajectory Score
    # -------------------------------------------------------------
    # 1. Growth component
    if len(career_history) < 2:
        growth_component = 0.5
    else:
        # Sort history chronologically
        jobs_chrono = []
        for job in career_history:
            if job:
                start = safe_date(job.get("start_date"))
                if start:
                    jobs_chrono.append((start, job))
        jobs_chrono.sort(key=lambda x: x[0])
        
        growth_events = 0
        valid_transitions = 0
        for i in range(len(jobs_chrono) - 1):
            j1 = jobs_chrono[i][1]
            j2 = jobs_chrono[i+1][1]
            size1_str = j1.get("company_size") or "201-500"
            size2_str = j2.get("company_size") or "201-500"
            
            t1 = SIZE_TIER_ORDER.get(size1_str, 3)
            t2 = SIZE_TIER_ORDER.get(size2_str, 3)
            
            # Growth event: starting company was <= 201-500 (index 3), and next was larger
            if t1 <= 3:
                valid_transitions += 1
                if t2 > t1:
                    growth_events += 1
                    
        if valid_transitions > 0:
            growth_component = growth_events / valid_transitions
        else:
            growth_component = 0.5
            
    # 2. Current component
    curr_size = profile.get("current_company_size") or "201-500"
    curr_tier = SIZE_TIER_ORDER.get(curr_size, 3)
    if curr_tier <= 1: # "1-10" or "11-50"
        current_component = 1.0
    elif curr_tier == 2: # "51-200"
        current_component = 0.5
    else:
        current_component = 0.0
        
    startup_trajectory_score = 0.6 * growth_component + 0.4 * current_component
    
    # -------------------------------------------------------------
    # Base Career Score
    # -------------------------------------------------------------
    career_score_base = (
        0.35 * company_type_score +
        0.28 * title_relevance_score +
        0.22 * stability_score +
        0.15 * startup_trajectory_score
    )
    
    # -------------------------------------------------------------
    # 5.8.5 Industry Relevance Multiplier
    # -------------------------------------------------------------
    relevant_ind_months = 0
    has_any_industry = False
    for job in career_history:
        if not job:
            continue
        ind = job.get("industry")
        duration = job.get("duration_months", 0) or 0
        if ind is not None:
            has_any_industry = True
            ind_lower = str(ind).lower()
            if any(term in ind_lower for term in INDUSTRY_RELEVANT_TERMS):
                relevant_ind_months += duration
                
    if has_any_industry and total_months > 0:
        industry_relevance_multiplier = 0.85 + (relevant_ind_months / total_months) * 0.25
    else:
        industry_relevance_multiplier = 1.0
        
    career_score = career_score_base * industry_relevance_multiplier
    
    # -------------------------------------------------------------
    # 5.9.2 CV / Speech / Robotics Domain-Mismatch Penalty
    # -------------------------------------------------------------
    skills = candidate.get("skills", []) or []
    cv_count = 0
    nlp_count = 0
    for s in skills:
        if s:
            name = (s.get("name") or "").lower().strip()
            if name in CV_SPEECH_ROBOTICS_SKILLS:
                cv_count += 1
            if name in NLP_IR_EVIDENCE_SKILLS:
                nlp_count += 1
                
    if cv_count >= 5 and nlp_count == 0:
        career_score *= 0.3
        
    return float(max(min(career_score, 1.0), 0.02))

def compute_experience_fit(candidate: dict) -> float:
    """
    Computes Experience Fit per Module 5.10.
    """
    y = float(candidate.get("profile", {}).get("years_of_experience", 0) or 0.0)
    if y < 3.0:
        return 0.10
    elif 3.0 <= y < 4.0:
        return 0.40
    elif 4.0 <= y < 5.0:
        return 0.65
    elif 5.0 <= y <= 9.0:
        return float(max(0.7, 1.0 - 0.15 * abs(y - 7.0)))
    elif 9.0 < y <= 12.0:
        return 0.70
    else:
        return 0.50

def compute_location_score(candidate: dict) -> float:
    """
    Computes Location Fit per Module 5.11.
    """
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower().strip()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    
    TIER_1 = {"pune", "noida", "delhi", "new delhi", "delhi ncr", "ncr", "gurgaon", "gurugram", "faridabad"}
    TIER_2 = {"hyderabad", "mumbai", "chennai"}
    
    location_tokens = set(re.split(r"[,/\s\-]+", location))
    
    if location_tokens.intersection(TIER_1):
        return 1.0
    elif location_tokens.intersection(TIER_2):
        return 0.80
    elif country == "india" and willing_to_relocate is True:
        return 0.65
    elif country == "india" and willing_to_relocate is False:
        return 0.30
    elif country != "india" and country != "" and willing_to_relocate is True:
        return 0.20
    else:
        return 0.00

def compute_education_score(candidate: dict) -> float:
    """
    Computes Education Tier score per Module 5.12.
    """
    education = candidate.get("education")
    if not education:
        return 0.35
        
    TIER_SCORES = {
        "tier_1": 1.00,
        "tier_2": 0.75,
        "tier_3": 0.50,
        "tier_4": 0.30,
        "unknown": 0.40
    }
    
    max_score = 0.0
    for edu in education:
        if edu:
            tier = edu.get("tier", "unknown")
            score = TIER_SCORES.get(tier, 0.40)
            if score > max_score:
                max_score = score
    return float(max_score)

def compute_engagement_score(candidate: dict) -> float:
    """
    Quantifies Recruiter Engagement per Module 6.2.
    """
    signals = candidate.get("redrob_signals", {}) or {}
    views = signals.get("profile_views_received_30d", 0) or 0
    saves = signals.get("saved_by_recruiters_30d", 0) or 0
    search = signals.get("search_appearance_30d", 0) or 0
    
    score = (
        min(int(views) / 50.0, 1.0) * 0.35 +
        min(int(saves) / 10.0, 1.0) * 0.35 +
        min(int(search) / 30.0, 1.0) * 0.30
    )
    return float(score)

def extract_features(candidate: dict) -> dict:
    """
    Legacy compatibility wrapper.
    """
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    
    # Distance from sweet spot
    if 5 <= yoe <= 9:
        yoe_distance = 0
    elif yoe < 5:
        yoe_distance = 5 - yoe
    else:
        yoe_distance = yoe - 9
        
    history = candidate.get("career_history", []) or []
    is_consulting_only = False
    if history:
        is_consulting_only = all(
            any(cfirm in (job.get("company") or "").lower() for cfirm in CONSULTING_FIRMS)
            for job in history if job
        )
        
    signals = candidate.get("redrob_signals", {}) or {}
    behavior_score = 1.0
    response_rate = signals.get("recruiter_response_rate", 0) or 0.0
    if response_rate < 0.3:
        behavior_score -= 0.2
        
    notice_period = signals.get("notice_period_days", 60) or 60
    if notice_period > 90:
        behavior_score -= 0.2
        
    # Check last active date
    last_active = signals.get("last_active_date", "")
    if last_active:
        last_active_dt = safe_date(last_active)
        if last_active_dt:
            days_inactive = (datetime.date(2026, 1, 1) - last_active_dt).days
            if days_inactive > 180:
                behavior_score -= 0.3

    behavior_score = max(0.5, behavior_score)
        
    return {
        "yoe": yoe,
        "yoe_distance": yoe_distance,
        "is_consulting_only": is_consulting_only,
        "behavior_score": behavior_score,
        "response_rate": response_rate,
        "notice_period": notice_period
    }