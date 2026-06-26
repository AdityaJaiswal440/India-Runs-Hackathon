# src/features/honeypot.py
import datetime
import math
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)

# TIER_1_SKILLS used in heuristic_extractor.py and precompute.py
TIER_1_SKILLS = {
    "embeddings", "embedding", "sentence-transformers", "sentence transformers",
    "sbert", "openai embeddings", "text-embedding-ada-002", "bge",
    "bge-large", "bge-m3", "e5", "multilingual-e5", "embedding drift",
    "index refresh", "reindexing",
    "vector database", "vector db", "vector store", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "elastic search", "solr", "faiss", "ann", "approximate nearest neighbor",
    "hybrid search", "hybrid retrieval", "retrieval", "rag",
    "retrieval augmented generation",
    "python",
    "ndcg", "mrr", "map", "precision@k", "recall@k",
    "evaluation framework", "ab testing", "a/b testing", "learning to rank",
    "ltr", "offline-to-online correlation"
}

# Tech inception dates for Rule 19
TECH_INCEPTION = {
    "langchain": (2022, 10),
    "qdrant": (2021, 6),
    "llama_2": (2023, 7),
    "llama 2": (2023, 7)
}

# Founding years for Rule 20
FOUNDING_YEARS = {
    "aws": 2006,
    "amazon web services": 2006,
    "azure": 2010,
    "gcp": 2008,
    "google cloud": 2008
}

# Indian and Foreign Cities for Rule 22
INDIAN_CITIES = {
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "ahmedabad",
    "chennai", "kolkata", "surat", "pune", "jaipur", "lucknow", "kanpur",
    "nagpur", "indore", "thane", "bhopal", "visakhapatnam", "patna",
    "vadodara", "ghaziabad", "ludhiana", "agra", "nashik", "faridabad",
    "meerut", "rajkot", "kalyan-dombivli", "vasai-virar", "varanasi",
    "srinagar", "aurangabad", "dhanbad", "amritsar", "navi mumbai",
    "allahabad", "howrah", "ranchi", "gwalior", "jabalpur", "coimbatore",
    "vijayawada", "jodhpur", "madurai", "raipur", "kota", "guwahati",
    "chandigarh", "solapur", "hubli-dharwad", "bareilly", "moradabad",
    "mysore", "gurgaon", "gurugram", "aligarh", "jalandhar", "tiruchirappalli",
    "bhubaneswar", "salem", "mira-bhayandar", "warangal", "guntur",
    "bhiwandi", "saharanpur", "gorakhpur", "bikaner", "amravati", "noida",
    "jamshedpur", "bhilai", "cuttack", "firozabad", "kochi", "nellore",
    "bhavnagar", "dehradun", "durgapur", "asansol", "rourkela", "nanded",
    "kolhapur", "ajmer", "akola", "gulbarga", "jamnagar", "uujain", "loni",
    "siliguri", "jhansi", "ulhasnagar", "jammu", "sangli-miraj", "belgaum",
    "mangalore", "ambattur", "tirunelveli", "malegaon", "gaya", "jalgaon",
    "udaipur", "maheshtala"
}

FOREIGN_CITIES = {
    "london", "new york", "san francisco", "tokyo", "paris", "berlin",
    "toronto", "vancouver", "seattle", "austin", "boston", "chicago",
    "los angeles", "sydney", "melbourne", "singapore", "dubai", "amsterdam",
    "dublin"
}

SENIOR_KEYWORDS = {"senior", "principal", "lead", "architect", "director", "head", "manager"}
JUNIOR_KEYWORDS = {"junior", "associate", "intern", "trainee", "fresher"}

def safe_date(date_str) -> datetime.date:
    """Parses a date string safely, returning None if malformed or missing."""
    if not date_str:
        return None
    try:
        if isinstance(date_str, datetime.date):
            return date_str
        if isinstance(date_str, datetime.datetime):
            return date_str.date()
        # Parse ISO string
        date_str = str(date_str).strip()
        return datetime.date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None

def get_cert_year(cert) -> int:
    """Extracts certification year safely from a cert dict or string."""
    if isinstance(cert, dict):
        year = cert.get("year")
        if year is not None:
            try:
                return int(year)
            except (ValueError, TypeError):
                pass
        date_val = cert.get("date")
        if date_val is not None:
            try:
                if isinstance(date_val, int):
                    return date_val
                date_str = str(date_val).strip()
                if date_str.isdigit():
                    return int(date_str)
                parsed = safe_date(date_str)
                if parsed:
                    return parsed.year
            except (ValueError, TypeError):
                pass
    elif isinstance(cert, str):
        match = re.search(r'\b(20\d{2}|19\d{2})\b', cert)
        if match:
            return int(match.group(1))
    return None

def is_honeypot(candidate: dict, today: datetime.date = None, clone_ids: set = None) -> bool:
    """
    Evaluates a candidate profile against 36 active honeypot rules.
    If ANY active rule matches, returns True (is a honeypot).
    """
    if today is None:
        today = datetime.date.today()
        
    if clone_ids is None:
        clone_ids = set()

    cid = candidate.get("candidate_id")
    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []
    career_history = candidate.get("career_history", []) or []
    certifications = candidate.get("certifications", []) or []
    redrob_signals = candidate.get("redrob_signals", {}) or {}

    # Rule 33: The Clone
    if cid and cid in clone_ids:
        return True

    try:
        # 1. Impossible Skill Duration
        total_career_months = sum(job.get("duration_months", 0) or 0 for job in career_history if job)
        for s in skills:
            if s and (s.get("duration_months", 0) or 0) > total_career_months:
                return True

        # 2. Platform Time Paradox
        signup_dt = safe_date(redrob_signals.get("signup_date"))
        last_active_dt = safe_date(redrob_signals.get("last_active_date"))
        if signup_dt and last_active_dt and signup_dt > last_active_dt:
            return True

        # 3. Salary Range Inversion
        expected_sal = redrob_signals.get("expected_salary_range_inr_lpa") or {}
        sal_min = expected_sal.get("min")
        sal_max = expected_sal.get("max")
        if sal_min is not None and sal_max is not None and sal_max < sal_min:
            return True

        # 4. Impossible Education Overlap
        valid_edu = []
        for edu in education:
            if edu and edu.get("start_year") is not None:
                valid_edu.append(edu)
        valid_edu.sort(key=lambda x: x.get("start_year"))
        for i in range(len(valid_edu) - 1):
            e1 = valid_edu[i]
            e2 = valid_edu[i+1]
            if e1.get("end_year") is not None and e2.get("start_year") is not None:
                if e1["end_year"] > e2["start_year"] + 1:
                    return True

        # 5. Suspicious Educational Chronology
        phd_ends = []
        bach_starts = []
        for edu in education:
            if edu:
                deg = (edu.get("degree") or "").lower()
                if "ph.d" in deg or "doctorate" in deg:
                    if edu.get("end_year") is not None:
                        phd_ends.append(edu["end_year"])
                if any(x in deg for x in ["b.tech", "b.e.", "b.sc", "bachelor"]):
                    if edu.get("start_year") is not None:
                        bach_starts.append(edu["start_year"])
        if phd_ends and bach_starts:
            if min(phd_ends) < max(bach_starts):
                return True

        # 6. Skill Inflation
        expert_zero_count = 0
        for s in skills:
            if s:
                prof = (s.get("proficiency") or "").lower().strip()
                dur = s.get("duration_months")
                if prof == "expert" and (dur is None or int(dur) == 0):
                    expert_zero_count += 1
        if expert_zero_count >= 4:
            return True

        # 7. Impossible Tenure
        for job in career_history:
            if job and job.get("is_current") is True:
                s_date = safe_date(job.get("start_date"))
                dur = job.get("duration_months", 0) or 0
                if s_date and s_date > datetime.date(2020, 1, 1) and dur > 96:
                    return True

        # 8. Timeline Contradiction
        if education:
            grad_years = [edu.get("end_year") for edu in education if edu and edu.get("end_year") is not None]
            if grad_years:
                earliest_grad = min(grad_years)
                sub_jobs = [job for job in career_history if job and (job.get("duration_months", 0) or 0) >= 6]
                if not sub_jobs:
                    sub_jobs = [job for job in career_history if job]
                
                job_start_years = []
                for job in sub_jobs:
                    s_date = safe_date(job.get("start_date"))
                    if s_date:
                        job_start_years.append(s_date.year)
                
                if job_start_years:
                    first_job_year = min(job_start_years)
                    if earliest_grad > first_job_year + 5:
                        return True

        # 9. Ghost Profile
        completeness = profile.get("completeness") or redrob_signals.get("profile_completeness_score") or 0
        open_to_work = profile.get("open_to_work") or redrob_signals.get("open_to_work_flag")
        resp_rate = redrob_signals.get("recruiter_response_rate")
        if completeness > 95 and open_to_work is False and resp_rate == 0.0 and last_active_dt:
            days_inactive = (today - last_active_dt).days
            if days_inactive > 365:
                return True

        # 10. Negative Tenure
        for job in career_history:
            if job:
                s_date = safe_date(job.get("start_date"))
                e_date = safe_date(job.get("end_date"))
                if s_date and e_date and s_date > e_date:
                    return True

        # 11. Age Limit Violation
        yoe = profile.get("years_of_experience")
        if yoe is not None and float(yoe) > 45.0:
            return True

        # 12. Multiple Current Jobs
        current_jobs_count = sum(1 for job in career_history if job and job.get("is_current") is True)
        if current_jobs_count > 1:
            return True

        # 13. YoE vs Career Mismatch
        if yoe is not None:
            sum_career_months = sum(job.get("duration_months", 0) or 0 for job in career_history if job)
            if abs((float(yoe) * 12.0) - sum_career_months) > 24.0:
                return True

        # Rule 14 is REMOVED.

        # 15. Offer Without Interview
        icr = redrob_signals.get("interview_completion_rate")
        oar = redrob_signals.get("offer_acceptance_rate")
        if icr is not None and oar is not None and icr == 0.0 and oar > 0.0:
            return True

        # 16. Zero-Time Response
        avg_resp = redrob_signals.get("avg_response_time_hours")
        if avg_resp is not None and float(avg_resp) == 0.0:
            return True

        # 17. Parallel Employment
        from collections import defaultdict
        comp_jobs = defaultdict(list)
        for job in career_history:
            if job:
                cname = (job.get("company") or "").lower().strip()
                if cname:
                    comp_jobs[cname].append(job)
        for cname, jobs in comp_jobs.items():
            if len(jobs) > 1:
                for i in range(len(jobs)):
                    for j in range(i + 1, len(jobs)):
                        j1 = jobs[i]
                        j2 = jobs[j]
                        s1 = safe_date(j1.get("start_date"))
                        e1 = safe_date(j1.get("end_date") or today)
                        s2 = safe_date(j2.get("start_date"))
                        e2 = safe_date(j2.get("end_date") or today)
                        if s1 and e1 and s2 and e2:
                            overlap_start = max(s1, s2)
                            overlap_end = min(e1, e2)
                            if overlap_end > overlap_start:
                                overlap_days = (overlap_end - overlap_start).days
                                if overlap_days > 90:
                                    return True

        # 18. Fake AI Expert (JD Trap)
        ai_skills_count = sum(1 for s in skills if s and (s.get("name") or "").lower().strip() in TIER_1_SKILLS)
        if ai_skills_count >= 5 and career_history:
            non_tech_keywords = {"hr", "human resources", "marketing", "sales", "recruiter", 
                                 "talent acquisition", "public relations", "pr", "event", 
                                 "content writer", "copywriter", "social media", "branding", 
                                 "customer support", "customer care"}
            all_non_tech = True
            for job in career_history:
                if job:
                    title = (job.get("title") or "").lower()
                    if not any(k in title for k in non_tech_keywords):
                        all_non_tech = False
                        break
            if all_non_tech:
                return True

        # 19. Tech Time Travel
        for s in skills:
            if s:
                sname = (s.get("name") or "").lower().strip()
                if sname in TECH_INCEPTION:
                    inc_yr, inc_mo = TECH_INCEPTION[sname]
                    tech_age_months = (today.year - inc_yr) * 12 + (today.month - inc_mo)
                    dur = s.get("duration_months", 0) or 0
                    if dur > tech_age_months:
                        return True

        # 20. Pre-Cognition Cert
        for cert in certifications:
            year = get_cert_year(cert)
            if year is not None:
                cname = ""
                if isinstance(cert, dict):
                    cname = (cert.get("name") or "").lower()
                elif isinstance(cert, str):
                    cname = cert.lower()
                for issuer, f_year in FOUNDING_YEARS.items():
                    if issuer in cname:
                        if year < f_year:
                            return True

        # 21. Future Cert
        for cert in certifications:
            year = get_cert_year(cert)
            if year is not None and year > today.year:
                return True

        # 22. Geo Paradox
        country = (profile.get("country") or "").lower().strip()
        location = (profile.get("location") or "").lower()
        if country and location:
            loc_tokens = set(re.split(r"[,/\s\-]+", location))
            if country != "india":
                if any(city in loc_tokens for city in INDIAN_CITIES):
                    return True
            else:
                if any(city in loc_tokens for city in FOREIGN_CITIES):
                    return True

        # 23. Unverified Active
        email_ver = redrob_signals.get("verified_email")
        phone_ver = redrob_signals.get("verified_phone")
        linkedin_ver = redrob_signals.get("linkedin_connected")
        if open_to_work is True and completeness > 90:
            if email_ver is False and phone_ver is False and linkedin_ver is False:
                return True

        # 24. 1-Year Bachelor's
        for edu in education:
            if edu:
                deg = (edu.get("degree") or "").lower()
                if any(x in deg for x in ["b.tech", "b.e.", "b.sc", "bachelor"]):
                    s_yr = edu.get("start_year")
                    e_yr = edu.get("end_year")
                    if s_yr is not None and e_yr is not None and (e_yr - s_yr) < 3:
                        return True

        # 25. 1-Year Ph.D.
        for edu in education:
            if edu:
                deg = (edu.get("degree") or "").lower()
                if any(x in deg for x in ["ph.d", "doctorate"]):
                    s_yr = edu.get("start_year")
                    e_yr = edu.get("end_year")
                    if s_yr is not None and e_yr is not None and (e_yr - s_yr) < 2:
                        return True

        # 26. Uniform Skill Dist
        if len(skills) >= 8:
            durations = [s.get("duration_months", 0) or 0 for s in skills if s]
            endorsements = [s.get("endorsements", 0) or 0 for s in skills if s]
            if len(durations) == len(skills) and len(endorsements) == len(skills):
                if all(d == durations[0] for d in durations) and all(e == endorsements[0] for e in endorsements):
                    return True

        # 27. Mass Applier Bot
        apps = redrob_signals.get("applications_submitted_30d")
        if apps is not None and int(apps) > 75:
            return True

        # 28. Duration Math
        for job in career_history:
            if job:
                s_date = safe_date(job.get("start_date"))
                e_date = safe_date(job.get("end_date") or today)
                dec_months = job.get("duration_months")
                if s_date and e_date and dec_months is not None:
                    comp_months = (e_date.year - s_date.year) * 12 + (e_date.month - s_date.month)
                    if abs(comp_months - int(dec_months)) > 2:
                        return True

        # 29. Career Regression
        senior_starts = []
        junior_starts = []
        for job in career_history:
            if job:
                s_date = safe_date(job.get("start_date"))
                if s_date:
                    title = (job.get("title") or "").lower()
                    if any(k in title for k in SENIOR_KEYWORDS):
                        senior_starts.append(s_date)
                    if any(k in title for k in JUNIOR_KEYWORDS):
                        junior_starts.append(s_date)
        for s_start in senior_starts:
            for j_start in junior_starts:
                if (j_start - s_start).days > 365:
                    return True

        # 30. Schema Bounds
        notice = redrob_signals.get("notice_period_days")
        github_score = redrob_signals.get("github_activity_score")
        if notice is not None and int(notice) > 180:
            return True
        if completeness is not None and float(completeness) > 100.0:
            return True
        if yoe is not None and float(yoe) > 50.0:
            return True
        if github_score is not None and float(github_score) < -1.0:
            return True

        # 31. Assessment Orphan
        assess_scores = redrob_signals.get("skill_assessment_scores") or {}
        if assess_scores:
            s_names = { (s.get("name") or "").lower().strip() for s in skills if s }
            for a_skill in assess_scores.keys():
                if a_skill.lower().strip() not in s_names:
                    return True

        # 32. Views/Saves Paradox
        views = redrob_signals.get("profile_views_received_30d")
        saves = redrob_signals.get("saved_by_recruiters_30d")
        if views is not None and saves is not None:
            if int(views) > 500 and int(saves) == 0:
                return True

        # 34. Endorsement Mismatch
        tot_endorsements = redrob_signals.get("endorsements_received")
        if tot_endorsements is not None:
            sum_skills_end = sum(s.get("endorsements", 0) or 0 for s in skills if s)
            if abs(sum_skills_end - int(tot_endorsements)) > 5:
                return True

        # 35. Negative Tenure (Current Job)
        for job in career_history:
            if job and job.get("is_current") is True:
                s_date = safe_date(job.get("start_date"))
                if s_date and s_date > today:
                    return True

        # 36. Assessment Out-of-Bounds
        for score in assess_scores.values():
            if score is not None:
                if float(score) < 0.0 or float(score) > 100.0:
                    return True

        # 37. Empty Profile w/ High Engagement
        if len(skills) == 0 and len(career_history) == 0 and len(education) == 0:
            search_app = redrob_signals.get("search_appearance_30d")
            views = redrob_signals.get("profile_views_received_30d")
            if search_app is not None and views is not None:
                if int(search_app) > 50 and int(views) > 50:
                    return True

    except Exception as e:
        logger.warning(f"Error in is_honeypot calculation: {e}. Defaulting to False.")
        return False

    return False