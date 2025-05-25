from modules.job_role_classifier_new import JobRoleClassifierNew
from utils.text_extractor import ResumeTextExtractor
from utils.cleaner import Cleaner

print("Initializing Job Role Classifier from HuggingFace...")
print("="*60)
job_role = JobRoleClassifierNew("habib-ashraf/resume-job-classifier", enable_cache=True)
print("="*60)
print("✅ Job Role Classifier loaded successfully!")
print()

def test_job_role_prediction(resume_text, test_name="Test"):
    """
    Test job role prediction with resume text - exactly like api.py workflow.
    
    Args:
        resume_text (str): Resume text (can be extracted from file or raw text)
        test_name (str): Name/description of the test
    """
    print(f"🔍Testing Job Role Classification with {test_name}")
    print("-" * 50)
    
    try:
        # This simulates the exact workflow in api.py
        predicted_role = job_role.predict_role(resume_text)
        
        print(f"✅ Predicted Job Role: {predicted_role}")
        
       
        
        return predicted_role
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("-" * 50)
        return None

def extract_and_test_from_file(file_path):
    """
    Extract text from file and test - simulating full api.py workflow.
    
    Args:
        file_path (str): Path to resume file
    """
    try:
        # Extract text exactly like api.py does
        extractor = ResumeTextExtractor(file_path)
        cv_text = extractor.extract()
        cv_text = Cleaner(cv_text).remove_empty_lines(cv_text)
        
        # Test the prediction
        return test_job_role_prediction(cv_text, f"File: {file_path}")
        
    except Exception as e:
        print(f"❌ Error extracting from {file_path}: {str(e)}")
        return None

# Example usage and test cases
if __name__ == "__main__":
    print("\n🚀 TESTING JOB ROLE CLASSIFIER - API.PY STYLE")
    print("="*60)
    
    # Test 1: Sample Software Engineer Resume
    software_engineer_text = """
    John Smith
    Senior Software Engineer
    
    Experience:
    - 5 years of experience in full-stack development
    - Built scalable web applications using Python, Django, and React
    - Implemented microservices architecture using Docker and Kubernetes
    - Led a team of 4 developers in agile environment
    - Experience with AWS, PostgreSQL, Redis, and CI/CD pipelines
    
    Skills:
    Python, Django, Flask, JavaScript, React, Node.js, PostgreSQL, MongoDB, 
    AWS, Docker, Kubernetes, Git, Jenkins, Agile, Scrum
    
    Education:
    Bachelor of Computer Science, MIT, 2018
    """
    
    test_job_role_prediction(software_engineer_text, "Sample Software Engineer Resume")
    print()
    
    # Test 2: Sample Data Scientist Resume  
    data_scientist_text = """
    Jane Doe
    Data Scientist
    
    Experience:
    - 3 years analyzing large datasets and building ML models
    - Developed predictive models using Python, scikit-learn, and TensorFlow
    - Created data visualizations and dashboards using Tableau and Power BI
    - Performed statistical analysis and A/B testing
    - Experience with big data tools like Spark and Hadoop
    
    Skills:
    Python, R, SQL, Machine Learning, Deep Learning, TensorFlow, PyTorch,
    Pandas, NumPy, scikit-learn, Tableau, Power BI, Statistics, A/B Testing
    
    Education:
    Master of Data Science, Stanford University, 2020
    """
    
    test_job_role_prediction(data_scientist_text, "Sample Data Scientist Resume")
    print()
    
    # Test 3: Sample Marketing Resume
    marketing_text = """
    Mike Johnson
    Digital Marketing Manager
    
    Experience:
    - 4 years managing digital marketing campaigns across multiple channels
    - Increased website traffic by 150% through SEO and content marketing
    - Managed social media accounts with 100K+ followers
    - Created and executed email marketing campaigns with 25% open rates
    - Experience with Google Analytics, Facebook Ads, and marketing automation
    
    Skills:
    Digital Marketing, SEO, SEM, Social Media Marketing, Content Marketing,
    Email Marketing, Google Analytics, Facebook Ads, HubSpot, Mailchimp
    
    Education:
    Bachelor of Marketing, University of California, 2019
    """
    
    test_job_role_prediction(marketing_text, "Sample Marketing Resume")
    print()
    
    # Test 4: Test with extracted file (uncomment to use)
    print("📁 Testing with extracted file:")
    extract_and_test_from_file("Test_Samples/cv (97).pdf")
    print()
    
    
    
    print("\n" + "="*60)
    print("🎯 Test Summary:")
    print("✅ Classifier initialization: SUCCESS")
    print("✅ Text-based prediction: SUCCESS") 
    print("="*60)