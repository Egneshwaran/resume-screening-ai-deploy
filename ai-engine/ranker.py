from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from gemini_service import GeminiMatcher

class CandidateRanker:
    def __init__(self, use_gemini: bool = False):
        self.vectorizer = TfidfVectorizer()
        self.gemini = GeminiMatcher() if use_gemini else None

    def calculate_similarity(self, job_desc, resume_text):
        if not job_desc or not resume_text:
            return 0.0
        try:
            # Ensure they are strings
            job_desc = str(job_desc)
            resume_text = str(resume_text)
            tfidf_matrix = self.vectorizer.fit_transform([job_desc, resume_text])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            print(f"Similarity Calculation Error: {e}")
            return 0.0

    def analyze_skill_gap(self, required_skills, candidate_skills):
        if not required_skills:
            return {"match_percentage": 0, "missing_skills": [], "matched_skills": []}
            
        required = set([s.strip().lower() for s in str(required_skills).split(",") if s.strip()])
        candidate = set([str(s).lower() for s in (candidate_skills or [])])
        
        missing = required - candidate
        match_pct = (len(required - missing) / len(required)) * 100 if required else 0
        
        return {
            "match_percentage": match_pct,
            "missing_skills": list(missing),
            "matched_skills": list(required & candidate)
        }
    
    def analyze_experience(self, required_range, candidate_years):
        # required_range e.g. "2-4 years", "5+ years", "0-1 years"
        # If None or empty, assume 0
        if not required_range:
            return 100.0
            
        min_exp = 0.0
        max_exp = 100.0
        
        # Check for "N-M years" or "N+ years"
        range_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', required_range)
        plus_match = re.search(r'(\d+)\+', required_range)
        
        if range_match:
            min_exp = float(range_match.group(1))
            max_exp = float(range_match.group(2))
        elif plus_match:
            min_exp = float(plus_match.group(1))
            max_exp = 100.0 # Open ended
        else:
            # Fallback if just a number is given "2 years"
            simple_match = re.search(r'(\d+)', required_range)
            if simple_match:
                min_exp = float(simple_match.group(1))
            else:
                try:
                     # Try to parse if it's just a number string "2"
                     min_exp = float(required_range)
                except:
                     min_exp = 0.0

        if candidate_years >= min_exp:
            return 100.0
        else:
            # Calculate gap
            if min_exp == 0: return 100.0
            score = (candidate_years / min_exp) * 100.0
            return max(0.0, min(100.0, score))

    def generate_explanation(self, analysis, similarity_score, exp_score, required_exp, candidate_exp):
        explanation = f"Skill match: {analysis['match_percentage']:.1f}%. "
        
        if required_exp:
            explanation += f"Experience: {candidate_exp} yrs (Req: {required_exp}). "
        
        if analysis['missing_skills']:
            explanation += f"Missing: {', '.join(analysis['missing_skills'])}. "
        
        explanation += f"Content relevance: {similarity_score * 100:.1f}%."
        return explanation
    
    def rank_candidates(self, job, resumes):
        results = []
        for resume in resumes:
            analysis = self.analyze_single(job, resume)
            results.append(analysis)
        
        return sorted(results, key=lambda x: x['total_score'], reverse=True)

    def analyze_single(self, job, resume):
        w_skill = job.get('skill_weight', 50) / 100.0
        w_exp = job.get('experience_weight', 30) / 100.0
        w_desc = job.get('description_weight', 20) / 100.0

        sim_score = self.calculate_similarity(job['description'], resume['text'])
        gap_analysis = self.analyze_skill_gap(job['required_skills'], resume.get('skills', []))
        candidate_exp = resume.get('experience_years', 0.0)
        exp_score = self.analyze_experience(job.get('required_experience', '0'), candidate_exp)
        
        total_score = (gap_analysis['match_percentage'] * w_skill) + (exp_score * w_exp) + (sim_score * 100 * w_desc)
        
        explanation = self.generate_explanation(gap_analysis, sim_score, exp_score, job.get('required_experience'), candidate_exp)
        return {
            "resume_id": resume.get('id', 0),
            "total_score": round(total_score, 2),
            "skill_score": round(gap_analysis['match_percentage'], 2),
            "experience_score": round(exp_score, 2),
            "explanation": explanation
        }
