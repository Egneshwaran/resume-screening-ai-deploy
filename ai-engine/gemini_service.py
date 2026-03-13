import os
import google.generativeai as genai
from typing import Dict, List, Optional
import json

class GeminiMatcher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def analyze_match(self, job_description: str, resume_text: str, 
                     required_skills: str = "", required_experience: str = "",
                     skill_weight: int = 50, exp_weight: int = 30, desc_weight: int = 20) -> Dict:
        if not self.model:
            return {"error": "Gemini API key not configured."}
        
        prompt = f"Calculate match score between resume and job. Weights: Skills={skill_weight}%, Experience={exp_weight}%, Description={desc_weight}%"

        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except Exception as e:
            return {"error": str(e)}
