from utils.text_extractor import ResumeTextExtractor
from modules.education_extractor import EducationExtractor
from modules.experience_classifier import ExperienceClassifier
from modules.skills_extractor import load_skills, extract_skills_from_text
from utils.cleaner import Cleaner
from modules.contact_extractor import extract_all_emails, extract_phone_number

resume_text = """John Doe
Email: john.doe@example.com
Phone: +1 (555) 123-4567
...
References available on request."""




# Example usage
file_path = "Test_Samples\Yusin-Resume.pdf"
extractor = ResumeTextExtractor(file_path)
text = extractor.extract()

resume_text = """
Jane Wanjiku earned her Bachelor's degree in Business Information Technology at Strathmore University.
She later completed her MBA in Project Management at the University of Nairobi.
"""
file_path = "modules\Education_DB_0.xlsx"
extractor = EducationExtractor(file_path)  # Automatically loads education_dataset.xlsx
results = extractor.extract_education_details(resume_text)

model_path = "model_exp1.pkl"
job_role = ExperienceClassifier(model_path, "tfidf_exp1.pkl", "encoder_exp1.pkl")
results = job_role.predict_role(resume_text)

# Load skills once
skills_list = load_skills('skills_database.csv')  # Adjust path as per your project

# Extract skills from a resume

matched_skills = extract_skills_from_text(text, skills_list)

# print("Matched skills:", matched_skills)


# print(results)
emails = extract_all_emails(text)
phone = extract_phone_number(text)


print("Email(s):", emails)
print("Phone:", phone)
