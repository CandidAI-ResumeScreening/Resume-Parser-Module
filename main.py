from utils.text_extractor import ResumeTextExtractor
from modules.education_extractor import EducationExtractor
from modules.experience_classifier import ExperienceClassifier
from modules.skills_extractor import load_skills, extract_skills_from_text
from utils.cleaner import Cleaner
from modules.contact_extractor import extract_all_emails_new, extract_phone_number
from modules.experience_predictor import ExperiencePredictor
from modules.language_extractor import LanguageExtractor
from pathlib import Path

# Initialize once with your Excel file
extractor_language = LanguageExtractor("utils\data\languages_Database.xlsx", column_name="Language")


# Initialize predictor with paths to your saved files

predictor = ExperiencePredictor(
    model_path=str(Path('models') / 'final_experience_model (1).pkl'),
    vectorizer_path=str(Path('models') / 'experience_vectorizer (1).pkl'),
    label_encoder_path=str(Path('models') / 'experience_label_encoder (1).pkl')
)


# Example usage
file_path = "Test_Samples\Yunus-Resume.pdf"
extractor = ResumeTextExtractor(file_path)
text = extractor.extract()


file_path = "utils\data\Education_DB_0.xlsx"
extractor = EducationExtractor(file_path)  # Automatically loads education_dataset.xlsx
job_role = ExperienceClassifier(
    str(Path("models") / "model_exp1.pkl"),
    str(Path("models") / "tfidf_exp1.pkl"),
    str(Path("models") / "encoder_exp1.pkl")
)
# Load skills once
skills_list = load_skills('utils\data\skills_database.csv')  # Adjust path as per your project

# Extract skills from a resume


# print("Matched skills:", matched_skills)


# print(text)
emails = extract_all_emails_new(text)
phone = extract_phone_number(text)
designation = job_role.predict_role(text)
exp_lvl =  predictor.predict_experience(text)
education_details = extractor.extract_education_details(text)
matched_skills = extract_skills_from_text(text, skills_list)
language_spoken = extractor_language.extract_languages(text)

candidate_details = {
    "Email": emails,
    "Phone Number": phone,
    "Job role": designation,
    "Experience level": exp_lvl,
    "Education info": education_details,
    "Skills": matched_skills,
    "languages spoken": language_spoken
}
print("-----------the Raw Candidate details extracted--------------------")
print(candidate_details)

print("-------------------------More readable format----------------------")
print("📄 Candidate Details:")
print("----------------------------")
for key, value in candidate_details.items():
    if key == "Education info" and isinstance(value, dict):
        print(f"{key}:")
        for sub_key, sub_value in value.items():
            formatted = ", ".join(sorted(sub_value)) if sub_value else "N/A"
            print(f"   • {sub_key}: {formatted}")
    elif isinstance(value, list):
        print(f"{key}: {', '.join(value) if value else 'N/A'}")
    else:
        print(f"{key}: {value if value else 'N/A'}")




