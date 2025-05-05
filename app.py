from utils.text_extractor import ResumeTextExtractor
from utils.cleaner import Cleaner
from modules.contact_extractor import extract_all_emails_new, extract_first_phone_number
from modules.experience_level_classifier import ExperienceLevelClassifier
from modules.job_role_classifier import JobRoleClassifier
import os
from openai import AzureOpenAI
from pathlib import Path
import json
from typing import Dict, List, Union
from dotenv import load_dotenv


load_dotenv()



# Use environment variables instead of hardcoding values
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")


# file_path = "Test_Samples\cv (97).pdf"
# extractor = ResumeTextExtractor(file_path)
# cv_text = extractor.extract()
# cv_text = Cleaner(cv_text).remove_empty_lines(cv_text)

# emails = extract_all_emails_new(cv_text)
# phone = extract_first_phone_number(cv_text)

# predictor = ExperienceLevelClassifier(
#     model_path=str(Path('models') / 'final_experience_model (1).pkl'),
#     vectorizer_path=str(Path('models') / 'experience_vectorizer (1).pkl'),
#     label_encoder_path=str(Path('models') / 'experience_label_encoder (1).pkl')
# )
# exp_lvl =  predictor.predict_experience(cv_text)

# # Initialize your job role classifier model
# job_role = JobRoleClassifier(
#     str(Path("models") / "model_exp1.pkl"),
#     str(Path("models") / "tfidf_exp1.pkl"),
#     str(Path("models") / "encoder_exp1.pkl")
# )



# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)



def parse_resume_with_ai(resume_text: str) -> Dict[str, Union[str, List, int]]:
    """Use Azure OpenAI to extract structured resume information"""
    prompt = f"""
    Analyze the following resume text and extract the specified information in EXACTLY the JSON format provided below.
    Return ONLY valid JSON output with no additional text or explanation.

    Required fields:
    - Name (full name)
    - Job Role/Designation (current or most recent)
    - Social Media (LinkedIn, GitHub, Portfolio, Medium - only return valid URLs or account username if specified)
    - Education Details (list of dictionaries with: education level, field of study, institution, grade level, date completed)
    - Total Estimated Years of Experience (calculate from dates specified under experience section or return "Not specified")
    - Experience Details (list of dictionaries with each dictionary containing: Industry Name(this is the name of the Company/institution) and Roles)
    - Skills (both technical and non-technical)
    - Certifications/Professional Qualifications/awards  (as list)

    **Normalization Guidelines:**
    - Expand abbreviations and acronyms to their full forms (e.g., "B.Tech" → "Bachelor of Technology", "B.Sc" → "Bachelor of Science", "ML" → "Machine Learning", "CS" → "Computer Science", "CT" → "Computer Technology").
    - Normalize technical terms and skills to their most commonly known names (e.g., "Python3" → "Python", "JS" → "JavaScript").
    - Ensure consistent naming for education levels, skills, certifications, and job titles.
    - For any missing or unspecified information, use "n/a" as the value.

    For any missing information, use "n/a" as the value.

    Resume Text:
    {resume_text}  # Truncate to avoid token limits

    Required JSON Output Format:
    {{
        "Name": "string",
        "Job Role": "string",
        "Social Media": ["url1", "url2"],
        "Education Details": [
            {{
                "education level": "string",
                "field of study": "string",
                "institution": "string",
                "grade level": "string",
                "date completed": "string"
            }}
        ],
        "Total Estimated Years of Experience": "float or string(if not specified)",
        "Experience Details": [
            {{
                "Industry Name": "string",
                "Roles": "string"
            }}
        ],
        "Skills": ["skill1", "skill2"],
        "Certification": ["cert1", "cert2"]
    }}
    """
    
    response = client.chat.completions.create(
        
        messages=[
            {"role": "system", "content": "You are an expert resume parser that extracts structured information from resumes. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        
        max_tokens=4096,
        temperature=0.1,  # Lower temperature for more consistent results
        top_p=1,
        model=deployment,
        response_format={"type": "json_object"}  # Ensure JSON output
    )
    
    try:
        # Extract the JSON content from the response
        json_str = response.choices[0].message.content
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return {"error": "Failed to parse resume"}

def process_resume(resume_text: str) -> Dict:
    """Complete processing pipeline"""
    # Extract text from file
    text = resume_text
    
    # Parse with AI
    parsed_data = parse_resume_with_ai(text)
    
    # # Add contact info 
    # parsed_data["Email"] = emails
    # parsed_data["Phone"] = phone
    # parsed_data["Experience level"] = exp_lvl
    # # Fallback for missing Job Role
    # if parsed_data.get("Job Role", "").strip().lower() == "n/a":
    #     predicted_role = job_role.predict_role(text)
    #     parsed_data["Job Role"] = predicted_role
    
    return parsed_data

if __name__ == "__main__":
    # Example usage
    cv_text = "cv_text"
    result = process_resume(cv_text)
    
    print("Extracted Resume Data:")
    print(json.dumps(result, indent=2))
    
    # Save to file
    with open('refined_parsed_resume_for_cv_97.json', 'w') as f:
        json.dump(result, f, indent=2)
