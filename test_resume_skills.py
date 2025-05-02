from modules.skill_extractor import SkillExtractor
from utils.text_extractor import ResumeTextExtractor
from utils.cleaner import Cleaner



def test_with_local_resume(pdf_path):
    # Initialize extractor
    extractor = SkillExtractor("./resume_skills_ner_model_v2")

    # Extract resume text
    text_extractor = ResumeTextExtractor(pdf_path)
    cv_text = text_extractor.extract()
    # cv_text = Cleaner(cv_text).remove_empty_lines(cv_text)
    
    # Extract skills
    skills = extractor.extract_skills(cv_text)
    
    print("\nExtracted Skills:")
    # for idx, skill in enumerate(skills, 1):
    #     print(f"{idx}. {skill}")
    print("\n")
    print(skills)
    print("\n")
    print(f"\nTotal skills found: {len(skills)}")

# Example usage
if __name__ == "__main__":
    resume_path = "Test_Samples\Yusin-Resume.pdf"  # Change to your resume path
    test_with_local_resume(resume_path)