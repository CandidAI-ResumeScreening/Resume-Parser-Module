import unittest
import os
import json
import tempfile
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary functions without affecting the API
from utils.text_extractor import ResumeTextExtractor
from utils.cleaner import Cleaner
from modules.contact_extractor import extract_all_emails_new, extract_first_phone_number
from modules.experience_level_classifier import ExperienceLevelClassifier
from modules.job_role_classifier import JobRoleClassifier
from modules.language_extractor import LanguageExtractor
from modules.skills_extractor import extract_skills_from_text
from utils.preprocessor import extract_social_media_links
from modules.name_fallback_extractor import extract_name_from_email
from app import parse_resume_with_ai
from testResume_skills_ner import test_with_huggingface_model

# Load environment variables
load_dotenv()

class ExtendedResumeParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        # Initialize models and components
        cls.predictor = ExperienceLevelClassifier(
            model_path=str(Path('models') / 'final_experience_model (1).pkl'),
            vectorizer_path=str(Path('models') / 'experience_vectorizer (1).pkl'),
            label_encoder_path=str(Path('models') / 'experience_label_encoder (1).pkl')
        )
        
        cls.job_role = JobRoleClassifier(
            str(Path("models") / "model_exp1.pkl"),
            str(Path("models") / "tfidf_exp1.pkl"),
            str(Path("models") / "encoder_exp1.pkl")
        )

    def process_resume_file(self, file_path):
        """Process a resume file using the same methodology as original test_resume_parser.py"""
        # Extract text from file
        extractor = ResumeTextExtractor(file_path)
        cv_text = extractor.extract()
        cv_text = Cleaner(cv_text).remove_empty_lines(cv_text)
        
        # Extract basic information
        emails = extract_all_emails_new(cv_text)
        phone = extract_first_phone_number(cv_text)
        exp_lvl = self.predictor.predict_experience(cv_text)
        
        # Preprocess text for AI parsing
        preprocessed_resume_text = Cleaner(cv_text).preprocess_resume_text(cv_text)
        
        # Parse with AI
        parsed_data = parse_resume_with_ai(preprocessed_resume_text)
        
        # Add extracted info
        parsed_data["Email"] = emails
        parsed_data["Phone"] = phone
        parsed_data["Experience level"] = exp_lvl
        parsed_data["rawResumeText"] = preprocessed_resume_text
        
        if parsed_data.get("Job Role", "").strip().lower() == "n/a":
            predicted_role = self.job_role.predict_role(cv_text)
            parsed_data["Job Role"] = predicted_role
            
        # Fallback for Name
        name = parsed_data.get("Name", "").strip().lower()
        if not name or name == "n/a":
            if emails:
                first_email = emails.split(',')[0].strip()
                try:
                    fallback_name = extract_name_from_email(first_email)
                    parsed_data["Name"] = fallback_name if fallback_name else "Not specified"
                except Exception:
                    parsed_data["Name"] = "Not specified"
            else:
                parsed_data["Name"] = "Not specified"
                
        return parsed_data

    def compare_json_outputs(self, actual, expected, tolerance=0.8):
        """
        Compare actual and expected JSON outputs with some tolerance for differences
        Returns a score between 0 and 1 indicating similarity
        """
        def flatten_json(data, prefix=""):
            result = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    result.update(flatten_json(value, f"{prefix}{key}."))
                elif isinstance(value, list):
                    if all(isinstance(item, dict) for item in value):
                        for i, item in enumerate(value):
                            result.update(flatten_json(item, f"{prefix}{key}[{i}]."))
                    else:
                        # For simple lists (like skills), we join them for comparison
                        result[f"{prefix}{key}"] = sorted([str(item).lower() for item in value])
                else:
                    result[f"{prefix}{key}"] = str(value).lower()
            return result
        
        flat_actual = flatten_json(actual)
        flat_expected = flatten_json(expected)
        
        # Count matches
        matches = 0
        total_fields = len(flat_expected)
        
        for key, expected_value in flat_expected.items():
            if key in flat_actual:
                actual_value = flat_actual[key]
                
                # For lists (like skills), calculate overlap
                if isinstance(expected_value, list) and isinstance(actual_value, list):
                    expected_set = set(expected_value)
                    actual_set = set(actual_value)
                    
                    if expected_set and actual_set:  # Avoid division by zero
                        overlap = len(expected_set.intersection(actual_set))
                        max_possible = max(len(expected_set), len(actual_set))
                        field_score = overlap / max_possible
                        matches += field_score
                    elif not expected_set and not actual_set:  # Both empty
                        matches += 1
                
                # For strings, use simple equality
                elif expected_value == actual_value:
                    matches += 1
                # Partial string match (useful for names, titles, etc.)
                elif isinstance(expected_value, str) and isinstance(actual_value, str):
                    if expected_value in actual_value or actual_value in expected_value:
                        matches += 0.5
        
        similarity_score = matches / total_fields if total_fields > 0 else 0
        return similarity_score
    
    def test_glen_ochieng_resume_parsing(self):
        """Test parsing of Glen Ochieng's software developer resume"""
        file_path = "Test_Samples/Resume - General 1.pdf"
        
        expected_output = {
            "Certification": ["n/a"],
            "Education Details": [{
                "date completed": "November 2025 (Expected)",
                "education level": "Bachelor of Science",
                "field of study": "Computer Science", 
                "grade level": "n/a",
                "institution": "Jomo Kenyatta University of Agriculture and Technology"
            }],
            "Email": "glenochieng045@gmail.com",
            "Experience Details": [
                {"Industry Name": "CodeDay Winter 2023 (Nairobi)", "Roles": "Developer for USSD app Hwrc"},
                {"Industry Name": "Frontend Mentor", "Roles": "Developer for Multi-Step Form project"}
            ],
            "Experience level": "Entry",
            "Job Role": "Software Developer",
            "Name": "Glen Ochieng",
            "Phone": None,
            "Skills": ["Python", "C#", "Kotlin", "JavaScript", "TypeScript", "Tailwind CSS", "C++", 
                      "Java", "PHP", "ASP.NET Core", "React", "NextJS", "Jetpack Compose", "Flask", 
                      "Godot", "Docker", "Gradle", "Git", "GitHub", "Visual Studio Code", 
                      "Android Studio", "PyCharm", "Visual Studio", "Rider", "IntelliJ IDEA", "English"],
            "Social Media": ["https://www.linkedin.com/in/glen-omondi-22b57a257", "https://github.com/Mirror83"],
            "Total Estimated Years of Experience": "Not specified"
        }
        
        # Process the resume
        actual_output = self.process_resume_file(file_path)
        
        # Compare results with tolerance
        similarity_score = self.compare_json_outputs(actual_output, expected_output)
        
        # Test passes if similarity is above threshold (80%)
        self.assertGreaterEqual(similarity_score, 0.8, 
                               f"Glen Ochieng resume parsing failed with similarity score: {similarity_score}")
        
        print(f"Glen Ochieng resume similarity score: {similarity_score:.2f}")

class ExtendedResumeParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        # Initialize models and components
        cls.predictor = ExperienceLevelClassifier(
            model_path=str(Path('models') / 'final_experience_model (1).pkl'),
            vectorizer_path=str(Path('models') / 'experience_vectorizer (1).pkl'),
            label_encoder_path=str(Path('models') / 'experience_label_encoder (1).pkl')
        )
        
        cls.job_role = JobRoleClassifier(
            str(Path("models") / "model_exp1.pkl"),
            str(Path("models") / "tfidf_exp1.pkl"),
            str(Path("models") / "encoder_exp1.pkl")
        )

    def process_resume_file(self, file_path):
        """Process a resume file using the same methodology as original test_resume_parser.py"""
        # Extract text from file
        extractor = ResumeTextExtractor(file_path)
        cv_text = extractor.extract()
        cv_text = Cleaner(cv_text).remove_empty_lines(cv_text)
        
        # Extract basic information
        emails = extract_all_emails_new(cv_text)
        phone = extract_first_phone_number(cv_text)
        exp_lvl = self.predictor.predict_experience(cv_text)
        
        # Preprocess text for AI parsing
        preprocessed_resume_text = Cleaner(cv_text).preprocess_resume_text(cv_text)
        
        # Parse with AI
        parsed_data = parse_resume_with_ai(preprocessed_resume_text)
        
        # Add extracted info
        parsed_data["Email"] = emails
        parsed_data["Phone"] = phone
        parsed_data["Experience level"] = exp_lvl
        parsed_data["rawResumeText"] = preprocessed_resume_text
        
        if parsed_data.get("Job Role", "").strip().lower() == "n/a":
            predicted_role = self.job_role.predict_role(cv_text)
            parsed_data["Job Role"] = predicted_role
            
        # Fallback for Name
        name = parsed_data.get("Name", "").strip().lower()
        if not name or name == "n/a":
            if emails:
                first_email = emails.split(',')[0].strip()
                try:
                    fallback_name = extract_name_from_email(first_email)
                    parsed_data["Name"] = fallback_name if fallback_name else "Not specified"
                except Exception:
                    parsed_data["Name"] = "Not specified"
            else:
                parsed_data["Name"] = "Not specified"
                
        return parsed_data

    def compare_json_outputs(self, actual, expected, tolerance=0.8):
        """
        Compare actual and expected JSON outputs with some tolerance for differences
        Returns a score between 0 and 1 indicating similarity
        """
        def flatten_json(data, prefix=""):
            result = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    result.update(flatten_json(value, f"{prefix}{key}."))
                elif isinstance(value, list):
                    if all(isinstance(item, dict) for item in value):
                        for i, item in enumerate(value):
                            result.update(flatten_json(item, f"{prefix}{key}[{i}]."))
                    else:
                        # For simple lists (like skills), we join them for comparison
                        result[f"{prefix}{key}"] = sorted([str(item).lower() for item in value])
                else:
                    result[f"{prefix}{key}"] = str(value).lower()
            return result
        
        flat_actual = flatten_json(actual)
        flat_expected = flatten_json(expected)
        
        # Count matches
        matches = 0
        total_fields = len(flat_expected)
        
        for key, expected_value in flat_expected.items():
            if key in flat_actual:
                actual_value = flat_actual[key]
                
                # For lists (like skills), calculate overlap
                if isinstance(expected_value, list) and isinstance(actual_value, list):
                    expected_set = set(expected_value)
                    actual_set = set(actual_value)
                    
                    if expected_set and actual_set:  # Avoid division by zero
                        overlap = len(expected_set.intersection(actual_set))
                        max_possible = max(len(expected_set), len(actual_set))
                        field_score = overlap / max_possible
                        matches += field_score
                    elif not expected_set and not actual_set:  # Both empty
                        matches += 1
                
                # For strings, use simple equality
                elif expected_value == actual_value:
                    matches += 1
                # Partial string match (useful for names, titles, etc.)
                elif isinstance(expected_value, str) and isinstance(actual_value, str):
                    if expected_value in actual_value or actual_value in expected_value:
                        matches += 0.5
        
        similarity_score = matches / total_fields if total_fields > 0 else 0
        return similarity_score
    
    def test_glen_ochieng_resume_parsing(self):
        """Test parsing of Glen Ochieng's software developer resume"""
        file_path = "Test_Samples/Resume - General 1.pdf"
        
        expected_output = {
            "Certification": ["n/a"],
            "Education Details": [{
                "date completed": "November 2025 (Expected)",
                "education level": "Bachelor of Science",
                "field of study": "Computer Science", 
                "grade level": "n/a",
                "institution": "Jomo Kenyatta University of Agriculture and Technology"
            }],
            "Email": "glenochieng045@gmail.com",
            "Experience Details": [
                {"Industry Name": "CodeDay Winter 2023 (Nairobi)", "Roles": "Developer for USSD app Hwrc"},
                {"Industry Name": "Frontend Mentor", "Roles": "Developer for Multi-Step Form project"}
            ],
            "Experience level": "Entry",
            "Job Role": "Software Developer",
            "Name": "Glen Ochieng",
            "Phone": None,
            "Skills": ["Python", "C#", "Kotlin", "JavaScript", "TypeScript", "Tailwind CSS", "C++", 
                      "Java", "PHP", "ASP.NET Core", "React", "NextJS", "Jetpack Compose", "Flask", 
                      "Godot", "Docker", "Gradle", "Git", "GitHub", "Visual Studio Code", 
                      "Android Studio", "PyCharm", "Visual Studio", "Rider", "IntelliJ IDEA", "English"],
            "Social Media": ["https://www.linkedin.com/in/glen-omondi-22b57a257", "https://github.com/Mirror83"],
            "Total Estimated Years of Experience": "Not specified"
        }
        
        # Process the resume
        actual_output = self.process_resume_file(file_path)
        
        # Compare results with tolerance
        similarity_score = self.compare_json_outputs(actual_output, expected_output)
        
        # Test passes if similarity is above threshold (80%)
        self.assertGreaterEqual(similarity_score, 0.8, 
                               f"Glen Ochieng resume parsing failed with similarity score: {similarity_score}")
        
        print(f"Glen Ochieng resume similarity score: {similarity_score:.2f}")

    def test_business_resume_parsing(self):
        """Test parsing of Business resume"""
        file_path = "Test_Samples/Business_Resume.docx.pdf"
        
        expected_output = {
            "Certification": ["Taylor Opportunity Program for Students Scholarship Recipient",
                             "President's List (3 semesters)", "Dean's List (3 semesters)"],
            "Education Details": [{
                "date completed": "May 2021",
                "education level": "Bachelor of Arts",
                "field of study": "Marketing",
                "grade level": "Major GPA: 3.50/4.00; Overall GPA: 3.65/4.00",
                "institution": "Southeastern Louisiana University"
            }],
            "Email": "first.last@selu.edu",
            "Experience Details": [
                {"Industry Name": "Louisiana Department of Labor", "Roles": "Marketing Intern"},
                {"Industry Name": "American Red Cross, Fundraising Committee", "Roles": "Chairman of Advertising/Public Relations"},
                {"Industry Name": "Olive Garden", "Roles": "Server"},
                {"Industry Name": "The Body Shop", "Roles": "Assistant Store Manager"}
            ],
            "Experience level": "Entry",
            "Job Role": "Server",
            "Name": "first last",
            "Phone": "985-111-1111",
            "Skills": ["Conversational Spanish", "Written Spanish", "Bloomberg Terminal", 
                      "Microsoft Office Suite", "Spanish", "English"],
            "Social Media": ["n/a"],
            "Total Estimated Years of Experience": "4.0"
        }
        
        # Process the resume
        actual_output = self.process_resume_file(file_path)
        
        # Compare results with tolerance
        similarity_score = self.compare_json_outputs(actual_output, expected_output)
        
        # Test passes if similarity is above threshold (80%)
        self.assertGreaterEqual(similarity_score, 0.8, 
                               f"Business resume parsing failed with similarity score: {similarity_score}")
        
        print(f"Business resume similarity score: {similarity_score:.2f}")


class YusinResumeParserTest(unittest.TestCase):
    """Separate test class for Yusin Resume parsing"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        # Initialize models and components
        cls.predictor = ExperienceLevelClassifier(
            model_path=str(Path('models') / 'final_experience_model (1).pkl'),
            vectorizer_path=str(Path('models') / 'experience_vectorizer (1).pkl'),
            label_encoder_path=str(Path('models') / 'experience_label_encoder (1).pkl')
        )
        
        cls.job_role = JobRoleClassifier(
            str(Path("models") / "model_exp1.pkl"),
            str(Path("models") / "tfidf_exp1.pkl"),
            str(Path("models") / "encoder_exp1.pkl")
        )

    def process_resume_file(self, file_path):
        """Process a resume file using the same methodology as original test_resume_parser.py"""
        # Extract text from file
        extractor = ResumeTextExtractor(file_path)
        cv_text = extractor.extract()
        cv_text = Cleaner(cv_text).remove_empty_lines(cv_text)
        
        # Extract basic information
        emails = extract_all_emails_new(cv_text)
        phone = extract_first_phone_number(cv_text)
        exp_lvl = self.predictor.predict_experience(cv_text)
        
        # Preprocess text for AI parsing
        preprocessed_resume_text = Cleaner(cv_text).preprocess_resume_text(cv_text)
        
        # Parse with AI
        parsed_data = parse_resume_with_ai(preprocessed_resume_text)
        
        # Add extracted info
        parsed_data["Email"] = emails
        parsed_data["Phone"] = phone
        parsed_data["Experience level"] = exp_lvl
        parsed_data["rawResumeText"] = preprocessed_resume_text
        
        if parsed_data.get("Job Role", "").strip().lower() == "n/a":
            predicted_role = self.job_role.predict_role(cv_text)
            parsed_data["Job Role"] = predicted_role
            
        # Fallback for Name
        name = parsed_data.get("Name", "").strip().lower()
        if not name or name == "n/a":
            if emails:
                first_email = emails.split(',')[0].strip()
                try:
                    fallback_name = extract_name_from_email(first_email)
                    parsed_data["Name"] = fallback_name if fallback_name else "Not specified"
                except Exception:
                    parsed_data["Name"] = "Not specified"
            else:
                parsed_data["Name"] = "Not specified"
                
        return parsed_data

    def compare_json_outputs(self, actual, expected, tolerance=0.8):
        """
        Compare actual and expected JSON outputs with some tolerance for differences
        Returns a score between 0 and 1 indicating similarity
        """
        def flatten_json(data, prefix=""):
            result = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    result.update(flatten_json(value, f"{prefix}{key}."))
                elif isinstance(value, list):
                    if all(isinstance(item, dict) for item in value):
                        for i, item in enumerate(value):
                            result.update(flatten_json(item, f"{prefix}{key}[{i}]."))
                    else:
                        # For simple lists (like skills), we join them for comparison
                        result[f"{prefix}{key}"] = sorted([str(item).lower() for item in value])
                else:
                    result[f"{prefix}{key}"] = str(value).lower()
            return result
        
        flat_actual = flatten_json(actual)
        flat_expected = flatten_json(expected)
        
        # Count matches
        matches = 0
        total_fields = len(flat_expected)
        
        for key, expected_value in flat_expected.items():
            if key in flat_actual:
                actual_value = flat_actual[key]
                
                # For lists (like skills), calculate overlap
                if isinstance(expected_value, list) and isinstance(actual_value, list):
                    expected_set = set(expected_value)
                    actual_set = set(actual_value)
                    
                    if expected_set and actual_set:  # Avoid division by zero
                        overlap = len(expected_set.intersection(actual_set))
                        max_possible = max(len(expected_set), len(actual_set))
                        field_score = overlap / max_possible
                        matches += field_score
                    elif not expected_set and not actual_set:  # Both empty
                        matches += 1
                
                # For strings, use simple equality
                elif expected_value == actual_value:
                    matches += 1
                # Partial string match (useful for names, titles, etc.)
                elif isinstance(expected_value, str) and isinstance(actual_value, str):
                    if expected_value in actual_value or actual_value in expected_value:
                        matches += 0.5
        
        similarity_score = matches / total_fields if total_fields > 0 else 0
        return similarity_score

    def test_yusin_resume_parsing(self):
        """Test parsing of Yusin Ali Adan's software engineer resume"""
        file_path = "Test_Samples/Yusin-Resume.pdf"
        
        expected_output = {
            "Certification": ["n/a"],
            "Education Details": [{
                "date completed": "December 2025",
                "education level": "Bachelor of Science",
                "field of study": "Computer Science",
                "grade level": "n/a",
                "institution": "Jomo Kenyatta University of Agriculture and Technology"
            }],
            "Email": "yunisaden3@gmail.com",
            "Experience Details": [
                {"Industry Name": "JHUB Africa", "Roles": "Software Engineer Intern"},
                {"Industry Name": "SafariTech", "Roles": "Full-stack Developer"},
                {"Industry Name": "Salaam Microfinance Bank", "Roles": "CMS Designer"},
                {"Industry Name": "Rayk Wellness Center LLC", "Roles": "Freelance Developer"}
            ],
            "Experience level": "Entry",
            "Job Role": "Software Engineer Intern",
            "Name": "Yusin Ali Adan",
            "Phone": "0798654423",
            "Skills": ["C++", "C", "Python", "Java", "JavaScript", "Go", "SQL", "React.js", "Node.js", 
                      "Next.js", "Tailwind CSS", "Git", "Linux", "Docker", "Azure", "AWS", "MySQL", 
                      "PostgreSQL", "English"],
            "Social Media": ["https://linkedin.com", "https://github.com"],
            "Total Estimated Years of Experience": "0.33"
        }
        
        # Process the resume
        actual_output = self.process_resume_file(file_path)
        
        # Compare results with tolerance
        similarity_score = self.compare_json_outputs(actual_output, expected_output)
        
        # Test passes if similarity is above threshold (80%)
        self.assertGreaterEqual(similarity_score, 0.8, 
                               f"Yusin resume parsing failed with similarity score: {similarity_score}")
        
        print(f"Yusin resume similarity score: {similarity_score:.2f}")

    def test_business_resume_parsing(self):
        """Test parsing of Business resume"""
        file_path = "Test_Samples/Business_Resume.docx.pdf"
        
        expected_output = {
            "Certification": ["Taylor Opportunity Program for Students Scholarship Recipient",
                             "President's List (3 semesters)", "Dean's List (3 semesters)"],
            "Education Details": [{
                "date completed": "May 2021",
                "education level": "Bachelor of Arts",
                "field of study": "Marketing",
                "grade level": "Major GPA: 3.50/4.00; Overall GPA: 3.65/4.00",
                "institution": "Southeastern Louisiana University"
            }],
            "Email": "first.last@selu.edu",
            "Experience Details": [
                {"Industry Name": "Louisiana Department of Labor", "Roles": "Marketing Intern"},
                {"Industry Name": "American Red Cross, Fundraising Committee", "Roles": "Chairman of Advertising/Public Relations"},
                {"Industry Name": "Olive Garden", "Roles": "Server"},
                {"Industry Name": "The Body Shop", "Roles": "Assistant Store Manager"}
            ],
            "Experience level": "Entry",
            "Job Role": "Server",
            "Name": "first last",
            "Phone": "985-111-1111",
            "Skills": ["Conversational Spanish", "Written Spanish", "Bloomberg Terminal", 
                      "Microsoft Office Suite", "Spanish", "English"],
            "Social Media": ["n/a"],
            "Total Estimated Years of Experience": "4.0"
        }
        
        # Process the resume
        actual_output = self.process_resume_file(file_path)
        
        # Compare results with tolerance
        similarity_score = self.compare_json_outputs(actual_output, expected_output)
        
        # Test passes if similarity is above threshold (80%)
        self.assertGreaterEqual(similarity_score, 0.8, 
                               f"Business resume parsing failed with similarity score: {similarity_score}")
        
        print(f"Business resume similarity score: {similarity_score:.2f}")

if __name__ == "__main__":
    unittest.main()