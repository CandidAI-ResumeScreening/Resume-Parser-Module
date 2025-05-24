import pandas as pd
import re
import os
from typing import List

def load_skills(skills_csv_path: str = None) -> List[str]:
    """
    Load skills from a CSV file with a column named 'skills'. 
    Uses a default path if no custom path is provided.
    """
    if skills_csv_path is None:
        skills_csv_path = os.path.join("utils", "data", "updated_skills_database.csv")

    if not os.path.exists(skills_csv_path):
        raise FileNotFoundError(f"Skills file not found at: {skills_csv_path}")

    skills_df = pd.read_csv(skills_csv_path)
    
    if 'skill' not in skills_df.columns:
        raise ValueError(f"'skill' column not found in CSV file at: {skills_csv_path}")

    # Normalize to lowercase, strip whitespaces
    skills = [skill.strip().lower() for skill in skills_df['skill'].dropna()]

    return skills

def extract_skills_from_text(text: str, skills_list: List[str]) -> List[str]:
    """
    Extract skills from resume text using simple pattern matching.
    """
    text = text.lower()

    # Tokenize text to single/multi-word phrases
    tokens = re.findall(r'\b[\w\-\+\.#]+\b', text)
    phrases = tokens + [' '.join(tokens[i:i+2]) for i in range(len(tokens) - 1)] + \
              [' '.join(tokens[i:i+3]) for i in range(len(tokens) - 2)]

    matched_skills = list(set(skill for skill in skills_list if skill in phrases))
    
    return matched_skills

if __name__ == "__main__":
    resume_text = """
    I am a Python developer with experience in Django, React, and cloud platforms like AWS and Azure. 
    I am skilled in teamwork, problem solving, and using tools like Figma, Trello, and Git.
    """

    skills_list = load_skills()  # Uses default path utils/data/updated_skills_database.csv
    extracted = extract_skills_from_text(resume_text, skills_list)

    print("Extracted Skills:", extracted)
    print("\n")