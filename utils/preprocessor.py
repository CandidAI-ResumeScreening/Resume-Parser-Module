# preprocessing.py
import re
import string

def clean_resume(resume_text):
    """
    Clean and preprocess resume text
    """
    # Remove URLs
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
    # Remove RT and cc
    resume_text = re.sub(r'RT|cc', ' ', resume_text)
    # Remove hashtags
    resume_text = re.sub(r'#\S+', ' ', resume_text)
    # Remove mentions
    resume_text = re.sub(r'@\S+', ' ', resume_text)
    # Remove non-ASCII chars
    resume_text = re.sub(r'[^\x00-\x7f]', ' ', resume_text)
    # Remove extra whitespace
    resume_text = re.sub(r'\s+', ' ', resume_text)
    # Remove punctuations
    resume_text = resume_text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    resume_text = re.sub(r'[0-9]+', ' ', resume_text)
    # Convert to lowercase
    resume_text = resume_text.lower()
    # Remove stop words (we'll let TF-IDF handle this)
    return resume_text

