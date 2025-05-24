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
from app import parse_resume_with_ai

# Load environment variables
load_dotenv()

class ResumeParserTest(unittest.TestCase):
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
        """Process a resume file similarly to the API but without calling it"""
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
        
        # Fallback for Job Role if needed
        if parsed_data.get("Job Role", "").strip().lower() == "n/a":
            predicted_role = self.job_role.predict_role(cv_text)
            parsed_data["Job Role"] = predicted_role
            
        # Fallback for Name
        name = parsed_data.get("Name", "").strip().lower()
        if not name or name == "n/a":
            if emails:
                first_email = emails.split(',')[0].strip()
                try:
                    from modules.name_fallback_extractor import extract_name_from_email
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
        # Flatten the JSON to compare field by field
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
                    # If one is substring of the other or Levenshtein distance is small
                    if expected_value in actual_value or actual_value in expected_value:
                        matches += 0.5
        
        similarity_score = matches / total_fields if total_fields > 0 else 0
        return similarity_score
    
    def test_developer_resume_parsing(self):
        """Test parsing of software developer resume"""
        # Path to test resume
        file_path = "Test_Samples/software-developer-resume-example.pdf"
        
        # Expected output (from the provided example)
        expected_output = {
            "Certification": ["n/a"],
            "Education Details": [{
                "date completed": "May 2012",
                "education level": "Bachelor of Science",
                "field of study": "Computer Science",
                "grade level": "n/a",
                "institution": "University of Delaware"
            }],
            "Email": "cynthia@beamjobs.com",
            "Experience Details": [
                {"Industry Name": "QuickBooks", "Roles": "Software Developer"},
                {"Industry Name": "AMR", "Roles": "Front-End Developer"},
                {"Industry Name": "Kelly", "Roles": "Help Desk Analyst"}
            ],
            "Experience level": "Entry",
            "Job Role": "Software Developer",
            "Name": "Cynthia Dwayne",
            "Phone": "(123) 456-7890",
            "Skills": ["Python", "Django", "SQL", "PostgreSQL", "MySQL", "Cloud", "Google Cloud Platform", 
                      "Amazon Web Services", "JavaScript", "React", "Redux", "Node.js", "Typescript", 
                      "HTML", "CSS", "CI/CD", "English"],
            "Social Media": ["https://linkedin.com/in/cynthia-dwayne", "https://github.com/cynthia-dwayne"],
            "Total Estimated Years of Experience": "11"
        }
        
        # Process the resume
        actual_output = self.process_resume_file(file_path)
        
        # Compare results with tolerance
        similarity_score = self.compare_json_outputs(actual_output, expected_output)
        
        # Test passes if similarity is above threshold (80%)
        self.assertGreaterEqual(similarity_score, 0.8, 
                               f"Developer resume parsing failed with similarity score: {similarity_score}")
        
        # Print diagnostic info
        print(f"Developer resume similarity score: {similarity_score:.2f}")

    def test_senior_engineer_resume_parsing(self):
        """Test parsing of senior engineer resume"""
        # Path to test resume
        file_path = "Test_Samples/sample-resume.txt"
        
        # Expected output (from the provided example)
        expected_output = {
            "Certification": ["n/a"],
            "Education Details": [
                {
                    "date completed": "2020",
                    "education level": "Master of Science",
                    "field of study": "Computer Science",
                    "grade level": "GPA: 3.8/4.0",
                    "institution": "Stanford University"
                },
                {
                    "date completed": "2018",
                    "education level": "Bachelor of Science",
                    "field of study": "Electrical Engineering",
                    "grade level": "GPA: 3.6/4.0",
                    "institution": "University of California"
                }
            ],
            "Email": "john.doe@email.com",
            "Experience Details": [
                {"Industry Name": "Google", "Roles": "Senior Software Engineer"},
                {"Industry Name": "Microsoft", "Roles": "Software Engineer Intern"}
            ],
            "Experience level": "Expert",
            "Job Role": "Senior Software Engineer",
            "Name": "John Doe",
            "Phone": "(123) 456-7890",
            "Skills": ["Python", "Java", "C++", "Django", "React", "TensorFlow", "Docker", "AWS", "Git", "English"],
            "Social Media": ["n/a"],
            "Total Estimated Years of Experience": "5"
        }
        
        # Process the resume
        actual_output = self.process_resume_file(file_path)
        
        # Compare results with tolerance
        similarity_score = self.compare_json_outputs(actual_output, expected_output)
        
        # Test passes if similarity is above threshold (80%)
        self.assertGreaterEqual(similarity_score, 0.8, 
                               f"Senior engineer resume parsing failed with similarity score: {similarity_score}")
        
        # Print diagnostic info
        print(f"Senior engineer resume similarity score: {similarity_score:.2f}")

if __name__ == "__main__":
    unittest.main()