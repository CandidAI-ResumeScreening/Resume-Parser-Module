from utils.text_extractor import ResumeTextExtractor
from modules.education_extractor import EducationExtractor
from utils.cleaner import Cleaner



# Example usage
file_path = "Test_Samples\Yusin-Resume.pdf"
extractor = ResumeTextExtractor(file_path)
text = extractor.extract()
clean = Cleaner(text)

resume_text = """
Jane Wanjiku earned her Bachelor's degree in Business Information Technology at Strathmore University.
She later completed her MBA in Project Management at the University of Nairobi.
"""
file_path = "modules\Education_DB_0.xlsx"
extractor = EducationExtractor(file_path)  # Automatically loads education_dataset.xlsx
results = extractor.extract_education_details(resume_text)
# results = clean.remove_empty_lines(results)


print(results)