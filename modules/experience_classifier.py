import joblib
import numpy as np
from utils.preprocessor import clean_resume  # ✅ local clean_resume

class ExperienceClassifier:
    def __init__(self, model_path, tfidf_path, label_encoder_path):
    
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        label_encoder = joblib.load(label_encoder_path)
        
        self.tfidf = tfidf
        self.model = model
        self.label_encoder = label_encoder

    def predict_role(self, resume_text):
        cleaned = clean_resume(resume_text)
        transformed = self.tfidf.transform([cleaned])
        prediction_encoded = self.model.predict(transformed)
        prediction = self.label_encoder.inverse_transform(prediction_encoded)[0]
        
        if prediction == "Java Developer":
            prediction = "Backend Developer"
        
        return prediction
