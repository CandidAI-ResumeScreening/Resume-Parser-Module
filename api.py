from flask import Flask, request, jsonify, render_template_string, redirect
from flask_cors import CORS
from utils.text_extractor import ResumeTextExtractor
from utils.cleaner import Cleaner
from utils.preprocessor import clean_resume_for_exp, extract_social_media_links
from modules.contact_extractor import extract_all_emails_new, extract_first_phone_number
from modules.experience_level_classifier import ExperienceLevelClassifier
from modules.job_role_classifier import JobRoleClassifier
from modules.job_role_classifier_new import JobRoleClassifierNew
from modules.language_extractor import LanguageExtractor
from modules.name_fallback_extractor import extract_name_from_email
from modules.skills_extractor import extract_skills_from_text
from app import parse_resume_with_ai
from testResume_skills_ner import test_with_huggingface_model
from pathlib import Path
import os
from openai import AzureOpenAI
import json
import tempfile
from dotenv import load_dotenv
# import imghdr  # For image type validation

load_dotenv()

app = Flask(__name__)
CORS(app)  # This allows all domains. You can restrict later if needed.

# Initialize components
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

lang_db_path = os.getenv("LANGUAGES_DB_PATH")
lang_column = os.getenv("LANGUAGES_DB_COLUMN", "Language")  # Optional fallback

extractor_language = LanguageExtractor(lang_db_path, column_name=lang_column)

predictor = ExperienceLevelClassifier(
    model_path=str(Path('models') / 'final_experience_model (1).pkl'),
    vectorizer_path=str(Path('models') / 'experience_vectorizer (1).pkl'),
    label_encoder_path=str(Path('models') / 'experience_label_encoder (1).pkl')
)

# Initialize old classifier (loads immediately)
job_role_old = JobRoleClassifier(
    str(Path("models") / "model_exp1.pkl"),
    str(Path("models") / "tfidf_exp1.pkl"),
    str(Path("models") / "encoder_exp1.pkl")
)

# Initialize new classifier (lazy loading - no download yet)
job_role_new = JobRoleClassifierNew("habib-ashraf/resume-job-classifier", enable_cache=False)

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Supported file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def validate_image(file_stream):
#     """Validate that the uploaded file is actually an image"""
#     header = file_stream.read(32)  # Read first 32 bytes to determine file type
#     file_stream.seek(0)  # Reset file pointer
#     return imghdr.what(None, header)

@app.route('/')
def home():
    return redirect('/test')

@app.route('/test', methods=['GET'])
def test_form():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resume Parser</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            pre { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Upload Resume</h1>
        <form action="/parse" method="post" enctype="multipart/form-data">
            <input type="file" name="resume" accept=".pdf,.docx,.txt,.jpg,.jpeg,.png" required>
            <button type="submit">Parse Resume</button>
        </form>
    </body>
    </html>
    '''

@app.route('/parse', methods=['POST'])
def parse_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400
    
    # Additional validation for image files
    # if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        
    #     if image_type not in ('jpeg', 'png', 'jpg'):
    #         return jsonify({"error": "Invalid image file"}), 400
    
    try:
        # Save the uploaded file temporarily with appropriate extension
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
            file.save(tmp_file.name)
            file_path = tmp_file.name

        # Process the file
        extractor = ResumeTextExtractor(file_path)
        cv_text = extractor.extract()
        cv_text = Cleaner(cv_text).remove_empty_lines(cv_text)
        
        emails = extract_all_emails_new(cv_text)
        phone = extract_first_phone_number(cv_text)
        exp_lvl = predictor.predict_experience(cv_text)
        preprocessed_resume_text = Cleaner(cv_text).preprocess_resume_text(cv_text)
        
        parsed_data = parse_resume_with_ai(preprocessed_resume_text)
        parsed_data["Email"] = emails
        parsed_data["Phone"] = phone
        parsed_data["Experience level"] = exp_lvl
        parsed_data["rawResumeText"] = preprocessed_resume_text

        # Cascading Job Role Fallback System
        # job_role_result = parsed_data.get("Job Role", "").strip().lower()

        # if job_role_result == "n/a" or not job_role_result:
        #     # print("🔄 Job role from AI is 'n/a', trying old classifier...")
            
        #     try:
        #         # Try old classifier first
        #         predicted_role = job_role_old.predict_role(cv_text)
                
        #         if predicted_role and predicted_role.lower() not in ["n/a", "not specified", ""]:
        #             parsed_data["Job Role"] = predicted_role
        #             # print(f"✅ Old classifier succeeded: {predicted_role}")
        #         else:
        #             # Old classifier failed, try new classifier (downloads models now)
        #             # print("🔄 Old classifier failed, trying new HuggingFace classifier...")
        #             predicted_role_new = job_role_new.predict_role(cv_text)
        #             parsed_data["Job Role"] = predicted_role_new
        #             # print(f"✅ New classifier result: {predicted_role_new}")
                    
        #     except Exception as e:
        #         print(f" Old classifier error: {str(e)}")
        #         print("Trying new HuggingFace classifier as final fallback...")
                
        #         try:
        #             predicted_role_new = job_role_new.predict_role(cv_text)
        #             parsed_data["Job Role"] = predicted_role_new
        #             # print(f"✅ New classifier result: {predicted_role_new}")
        #         except Exception as e2:
        #             print(f"New classifier also failed: {str(e2)}")
        #             parsed_data["Job Role"] = "Not specified"

        #  Use ONLY the new classifier
        job_role_result = parsed_data.get("Job Role", "").strip().lower()

        if job_role_result == "n/a" or not job_role_result:
            # print("🔄 Job role from AI is 'n/a', trying new HuggingFace classifier...")
            
            try:
                # Use ONLY the new classifier (no old classifier fallback)
                predicted_role_new = job_role_new.predict_role(cv_text)
                parsed_data["Job Role"] = predicted_role_new
                print(f"✅ New classifier result: {predicted_role_new}")
                    
            except Exception as e:
                print(f"❌ New classifier error: {str(e)}")
                parsed_data["Job Role"] = "Error: Not Specified"
                
        # Fallback for Social Media links
        social_media = parsed_data.get("Social Media", [])

        # Check if Social Media is empty or contains only "n/a"
        if (not social_media or 
            social_media == ["n/a"] or 
            (len(social_media) == 1 and social_media[0].lower() in ["n/a", "not specified", ""])):
            
            print("Social Media is empty/n/a, extracting links from resume text...")
            
            try:
                # Extract social media links from resume text
                extracted_links = extract_social_media_links(cv_text)
                
                if extracted_links:
                    parsed_data["Social Media"] = extracted_links
                    print(f"Found {len(extracted_links)} social media links")
                    for i, link in enumerate(extracted_links[:3], 1):  # Show first 3 links
                        print(f"   {i}. {link}")
                    if len(extracted_links) > 3:
                        print(f"   ... and {len(extracted_links) - 3} more")
                else:
                    parsed_data["Social Media"] = []
                    print(" No social media links found in resume text")
                    
            except Exception as e:
                print(f" Error extracting social media links: {str(e)}")
                parsed_data["Social Media"] = []

        # Fallback for Name
        name = parsed_data.get("Name", "").strip().lower()

        # Fallback for Name
        if not name or name == "n/a":
            try:
                if emails:
                    # Handle comma-separated string of emails
                    first_email = emails.split(',')[0].strip()
                    fallback_name = extract_name_from_email(first_email)
                    parsed_data["Name"] = fallback_name if fallback_name else "Not specified"
                else:
                    parsed_data["Name"] = "Not specified"
            except Exception as e:
                # Add error handling to prevent crashes
                print(f"Error extracting name from email: {str(e)}")
                parsed_data["Name"] = "Not specified"
        
        # Extract spoken languages from resume text
        language_spoken = extractor_language.extract_languages(cv_text)
        skills = parsed_data.get("Skills", [])

        # If skills are empty or contain only 'n/a', apply fallback
        if not skills or (isinstance(skills, list) and len(skills) == 1 and skills[0].strip().lower() == "n/a"):
            try:
                # Fallback 1: Use extract_skills_from_text
                extracted_skills = extract_skills_from_text(cv_text)

                # If fallback 1 returns fewer than 2 skills, use fallback 2
                if len(extracted_skills) < 2:
                    extracted_skills = test_with_huggingface_model(file_path)

                parsed_data["Skills"] = extracted_skills

            except Exception as e:
                print(f"Skill extraction fallback error: {str(e)}")
                parsed_data["Skills"] = []

        # Merge spoken languages into skills (case-insensitive)
        skills_lower = [skill.lower() for skill in parsed_data["Skills"]]
        languages_to_add = [lang for lang in language_spoken if lang.lower() not in skills_lower]
        parsed_data["Skills"] += languages_to_add


        return jsonify(parsed_data)

    except Exception as e:
        return jsonify({"error": f"Error processing resume: {str(e)}"}), 500
    finally:
        if 'file_path' in locals():
            try:
                os.unlink(file_path)
            except:
                pass



if __name__ == '__main__':
    app.run(debug=True, port=5000)