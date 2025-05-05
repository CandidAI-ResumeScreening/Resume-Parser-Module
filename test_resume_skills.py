from modules.skill_extractor import SkillExtractor
from utils.text_extractor import ResumeTextExtractor
from utils.cleaner import Cleaner

def test_with_huggingface_model(pdf_path):
    # Initialize extractor with HuggingFace model
    model_name = "habib-ashraf/resume-skills-ner-v2"  # My uploaded HuggingFace model path
    print(f"Loading model from HuggingFace: {model_name}")
    extractor = SkillExtractor(model_name)

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
    resume_path = "Test_Samples/Yusin-Resume.pdf"  # Change to your resume path
    test_with_huggingface_model(resume_path)