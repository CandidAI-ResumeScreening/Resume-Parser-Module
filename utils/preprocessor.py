# preprocessing.py
import re
import string
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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


# Function to clean text
def clean_resume_for_exp(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)  # remove URLs
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

