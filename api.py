from flask import Flask, request, jsonify, render_template_string, redirect
from flask_cors import CORS
from utils.text_extractor import ResumeTextExtractor
from utils.cleaner import Cleaner
from modules.contact_extractor import extract_all_emails_new, extract_first_phone_number
from modules.experience_level_classifier import ExperienceLevelClassifier
from modules.job_role_classifier import JobRoleClassifier
from modules.language_extractor import LanguageExtractor
from app import parse_resume_with_ai
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

job_role = JobRoleClassifier(
    str(Path("models") / "model_exp1.pkl"),
    str(Path("models") / "tfidf_exp1.pkl"),
    str(Path("models") / "encoder_exp1.pkl")
)

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
        
        parsed_data = parse_resume_with_ai(cv_text)
        parsed_data["Email"] = emails
        parsed_data["Phone"] = phone
        parsed_data["Experience level"] = exp_lvl
        parsed_data["rawResumeText"] = cv_text
        
        if parsed_data.get("Job Role", "").strip().lower() == "n/a":
            predicted_role = job_role.predict_role(cv_text)
            parsed_data["Job Role"] = predicted_role
        
        # Extract spoken languages from resume text
        language_spoken = extractor_language.extract_languages(cv_text)
        skills = parsed_data.get("Skills", [])

        # Convert both lists to lowercase for comparison
        skills_lower = [skill.lower() for skill in skills]
        languages_to_add = [lang for lang in language_spoken if lang.lower() not in skills_lower]

        # Merge languages into the skills list
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
