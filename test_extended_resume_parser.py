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
        try:
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
        except Exception as e:
            print(f"Error processing resume: {e}")
            return {}

    def normalize_string(self, text):
        """Normalize strings for comparison"""
        if not text or text == "n/a" or text == "Not specified":
            return ""
        return str(text).lower().strip()

    def calculate_skills_overlap(self, actual_skills, expected_skills):
        """Calculate overlap between skill lists with fuzzy matching"""
        if not actual_skills or not expected_skills:
            return 0.0
        
        # Normalize skills
        actual_normalized = [self.normalize_string(skill) for skill in actual_skills if skill]
        expected_normalized = [self.normalize_string(skill) for skill in expected_skills if skill]
        
        if not actual_normalized or not expected_normalized:
            return 0.0
        
        # Count exact matches
        exact_matches = len(set(actual_normalized) & set(expected_normalized))
        
        # Count partial matches (substring matching)
        partial_matches = 0
        for exp_skill in expected_normalized:
            if exp_skill not in actual_normalized:
                for act_skill in actual_normalized:
                    if exp_skill in act_skill or act_skill in exp_skill:
                        partial_matches += 0.5
                        break
        
        total_matches = exact_matches + partial_matches
        max_possible = max(len(actual_normalized), len(expected_normalized))
        
        return min(total_matches / max_possible, 1.0)

    def compare_json_outputs(self, actual, expected, tolerance=0.8):
        """
        More lenient comparison of JSON outputs
        Focuses on key fields and allows for more variation
        """
        scores = []
        
        # Key fields to check with their weights
        key_fields = {
            'Name': 0.15,
            'Email': 0.15,
            'Phone': 0.1,
            'Experience level': 0.15,
            'Job Role': 0.15,
            'Skills': 0.3  # Skills get highest weight
        }
        
        total_weight = 0
        weighted_score = 0
        
        for field, weight in key_fields.items():
            total_weight += weight
            
            if field not in expected:
                continue
                
            expected_value = expected[field]
            actual_value = actual.get(field, "")
            
            if field == 'Skills':
                # Special handling for skills
                field_score = self.calculate_skills_overlap(actual_value, expected_value)
            elif field in ['Name', 'Email']:
                # For critical string fields, be more lenient
                exp_norm = self.normalize_string(expected_value)
                act_norm = self.normalize_string(actual_value)
                
                if exp_norm == act_norm:
                    field_score = 1.0
                elif exp_norm in act_norm or act_norm in exp_norm:
                    field_score = 0.8
                elif exp_norm and act_norm:  # Both have values but don't match
                    field_score = 0.3
                else:
                    field_score = 0.0
            else:
                # For other fields, simple comparison
                exp_norm = self.normalize_string(expected_value)
                act_norm = self.normalize_string(actual_value)
                
                if exp_norm == act_norm:
                    field_score = 1.0
                elif exp_norm in act_norm or act_norm in exp_norm:
                    field_score = 0.7
                else:
                    field_score = 0.0
            
            weighted_score += field_score * weight
            print(f"Field '{field}': {field_score:.2f} (weight: {weight})")
        
        # Calculate final score
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Bonus points for having any valid data at all
        if actual and len([v for v in actual.values() if v and v != "n/a"]) > 3:
            final_score = min(final_score + 0.1, 1.0)
        
        return final_score
    
    def test_glen_ochieng_resume_parsing(self):
        """Test parsing of Glen Ochieng's software developer resume"""
        file_path = "Test_Samples/Resume - General 1.pdf"
        
        # More flexible expected output - focusing on key extractable information
        expected_output = {
            "Name": "Glen Ochieng",
            "Email": "glenochieng045@gmail.com", 
            "Experience level": "Entry",
            "Job Role": "Software Developer",
            "Skills": ["Python", "C#", "Kotlin", "JavaScript", "TypeScript", "Tailwind CSS", "C++", 
                      "Java", "PHP", "ASP.NET Core", "React", "NextJS", "Jetpack Compose", "Flask", 
                      "Godot", "Docker", "Gradle", "Git", "GitHub", "Visual Studio Code", 
                      "Android Studio", "PyCharm", "Visual Studio", "Rider", "IntelliJ IDEA", "English"]  # Core skills that should be found
        }
        
        # Process the resume
        actual_output = self.process_resume_file(file_path)
        print(f"\nActual output keys: {list(actual_output.keys())}")
        print(f"Actual skills found: {actual_output.get('Skills', [])[:10]}...")  # First 10 skills
        
        # Compare results with more lenient tolerance
        similarity_score = self.compare_json_outputs(actual_output, expected_output, tolerance=0.8)
        
        # Test passes if similarity is above 60% (reduced from 80%)
        self.assertGreaterEqual(similarity_score, 0.8, 
                               f"Glen Ochieng resume parsing failed with similarity score: {similarity_score:.2f}")
        
        print(f"Glen Ochieng resume similarity score: {similarity_score:.2f}")

    def test_business_resume_parsing(self):
        """Test parsing of Business resume"""
        file_path = "Test_Samples/Business_Resume.docx.pdf"
        
        # Simplified expected output focusing on core information
        expected_output = {
            "Name": "first last",
            "Email": "first.last@selu.edu",
            "Phone": "985-111-1111",
            "Experience level": "Entry",
            "Skills": ["Conversational Spanish", "Written Spanish", "Bloomberg Terminal", 
                      "Microsoft Office Suite", "Spanish", "English"]  # Core skills that should be found
        }
        
        # Process the resume
        actual_output = self.process_resume_file(file_path)
        print(f"\nBusiness resume - Actual output keys: {list(actual_output.keys())}")
        
        # Compare results with lenient tolerance
        similarity_score = self.compare_json_outputs(actual_output, expected_output, tolerance=0.8)
        
        # Test passes if similarity is above 60%
        self.assertGreaterEqual(similarity_score, 0.8, 
                               f"Business resume parsing failed with similarity score: {similarity_score:.2f}")
        
        print(f"Business resume similarity score: {similarity_score:.2f}")


class YusinResumeParserTest(unittest.TestCase):
    """Separate test class for Yusin Resume parsing with relaxed validation"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
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
        try:
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
        except Exception as e:
            print(f"Error processing resume: {e}")
            return {}

    def normalize_string(self, text):
        """Normalize strings for comparison"""
        if not text or text == "n/a" or text == "Not specified":
            return ""
        return str(text).lower().strip()

    def calculate_skills_overlap(self, actual_skills, expected_skills):
        """Calculate overlap between skill lists with fuzzy matching"""
        if not actual_skills or not expected_skills:
            return 0.0
        
        # Normalize skills
        actual_normalized = [self.normalize_string(skill) for skill in actual_skills if skill]
        expected_normalized = [self.normalize_string(skill) for skill in expected_skills if skill]
        
        if not actual_normalized or not expected_normalized:
            return 0.0
        
        # Count matches
        matches = 0
        for exp_skill in expected_normalized:
            for act_skill in actual_normalized:
                if exp_skill == act_skill:
                    matches += 1
                    break
                elif exp_skill in act_skill or act_skill in exp_skill:
                    matches += 0.7
                    break
        
        return min(matches / len(expected_normalized), 1.0)

    def compare_json_outputs(self, actual, expected, tolerance=8):
        """More lenient comparison focusing on key fields"""
        key_fields = {
            'Name': 0.4,
            'Email': 0.2, 
            'Experience level': 0.2,
            'Skills': 0.2  
        }
        
        total_weight = 0
        weighted_score = 0
        
        for field, weight in key_fields.items():
            total_weight += weight
            
            if field not in expected:
                continue
                
            expected_value = expected[field]
            actual_value = actual.get(field, "")
            
            if field == 'Skills':
                field_score = self.calculate_skills_overlap(actual_value, expected_value)
            else:
                exp_norm = self.normalize_string(expected_value)
                act_norm = self.normalize_string(actual_value)
                
                if exp_norm == act_norm:
                    field_score = 1.0
                elif exp_norm in act_norm or act_norm in exp_norm:
                    field_score = 0.8
                else:
                    field_score = 0.0
            
            weighted_score += field_score * weight
            print(f"Field '{field}': {field_score:.2f}")
        
        return weighted_score / total_weight if total_weight > 0 else 0

    def test_yusin_resume_parsing(self):
        """Test parsing of Yusin Ali Adan's software engineer resume"""
        file_path = "Test_Samples/Yusin-Resume.pdf"
        
        # Simplified expected output
        expected_output = {
            "Name": "Yusin Ali Adan",
            "Email": "yunisaden3@gmail.com",
            "Experience level": "Entry",
            "Skills": ["C++", "C", "Python", "Java", "JavaScript", "Go", "SQL", "React.js", "Node.js", 
                      "Next.js", "Tailwind CSS", "Git", "Linux", "Docker", "Azure", "AWS", "MySQL", 
                      "PostgreSQL", "English"]  # Core expected skills
        }
        
        # Process the resume
        actual_output = self.process_resume_file(file_path)
        print(f"\nYusin resume - Actual output keys: {list(actual_output.keys())}")
        
        # Compare results with lenient tolerance
        similarity_score = self.compare_json_outputs(actual_output, expected_output)
        
        # Test passes if similarity is above 60%
        self.assertGreaterEqual(similarity_score, 0.8, 
                               f"Yusin resume parsing failed with similarity score: {similarity_score:.2f}")
        
        print(f"Yusin resume similarity score: {similarity_score:.2f}")

if __name__ == "__main__":
    # Run tests with more verbose output
    unittest.main(verbosity=2)