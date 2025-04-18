import re
import string
import nltk
import joblib
from nltk.corpus import stopwords
from utils.preprocessor import clean_resume_for_exp


from sklearn.preprocessing import LabelEncoder

# Download stopwords if not already done
nltk.download('stopwords')

class ExperienceLevelClassifier:
    def __init__(self, model_path, vectorizer_path, label_encoder_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#','', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ''.join([word for word in text if word not in string.punctuation])
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return " ".join(words)

    def predict_experience(self, resume_text):
        cleaned = self.clean_text(resume_text)
        vector = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vector)[0]
        return self.label_encoder.inverse_transform([pred])[0]
