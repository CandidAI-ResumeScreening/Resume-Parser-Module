import os
import joblib
import requests
from io import BytesIO
from pathlib import Path
from utils.preprocessor import clean_resume

class JobRoleClassifierNew:
    def __init__(self, model_name_or_path, enable_cache=False):
        """
        Initialize the job role classifier with lazy loading.
        Models are only downloaded when predict_role() is first called.
        
        Args:
            model_name_or_path (str): HuggingFace model name/path
            enable_cache (bool): Whether to enable local caching (default: False)
        """
        self.model_name_or_path = model_name_or_path
        self.enable_cache = enable_cache
        
        # Lazy loading - models not loaded until needed
        self.model = None
        self.tfidf = None
        self.label_encoder = None
        self._models_loaded = False
        
        if self.enable_cache:
            self.cache_dir = Path("./cache")
            self.cache_dir.mkdir(exist_ok=True)
            self._ensure_gitignore()

    def _load_models_if_needed(self):
        """
        Load models only when needed (lazy loading).
        This prevents unnecessary downloads unless the classifier is actually used.
        """
        if self._models_loaded:
            return  # Already loaded
            
        try:
            print("🔄 Loading job role classifier from HuggingFace (fallback mode)...")
            
            if self.enable_cache:
                # Load with caching
                self.model = self._load_with_cache("model_exp1.pkl")
                self.tfidf = self._load_with_cache("tfidf_exp1.pkl") 
                self.label_encoder = self._load_with_cache("encoder_exp1.pkl")
            else:
                # Load directly without caching
                self.model = self._download_and_load_pickle("model_exp1.pkl")
                self.tfidf = self._download_and_load_pickle("tfidf_exp1.pkl") 
                self.label_encoder = self._download_and_load_pickle("encoder_exp1.pkl")
            
            self._models_loaded = True
            print("✅ Job role classifier loaded successfully (fallback ready)")
            
        except Exception as e:
            print(f"❌ Error loading job role classifier: {str(e)}")
            raise

    def predict_role(self, resume_text):
        """
        Predict job role from resume text.
        Models are loaded on first call (lazy loading).
        
        Args:
            resume_text (str): Text extracted from a resume
            
        Returns:
            str: Predicted job role
        """
        try:
            # Load models only when prediction is actually needed
            self._load_models_if_needed()
            
            # Clean the resume text using the same preprocessing as before
            cleaned = clean_resume(resume_text)
            
            # Transform the text using the TF-IDF vectorizer
            transformed = self.tfidf.transform([cleaned])
            
            # Make prediction
            prediction_encoded = self.model.predict(transformed)
            
            # Decode the prediction
            prediction = self.label_encoder.inverse_transform(prediction_encoded)[0]
            
            # Apply the same fallback logic as the original
            if prediction == "Java Developer":
                prediction = "Backend Developer"
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error predicting job role: {str(e)}")
            return "Not specified"  # Fallback value

    def _ensure_gitignore(self):
        """
        Ensure the cache directory is in .gitignore to prevent accidentally 
        committing large model files to Git.
        """
        gitignore_path = Path('.gitignore')
        cache_pattern = f"{self.cache_dir}/"
        
        # Create .gitignore if it doesn't exist
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write(f"# Model cache directory (large files)\n{cache_pattern}\n")
            print(f"✅ Created .gitignore with cache directory: {cache_pattern}")
            return
        
        # Check if cache directory is already in .gitignore
        try:
            with open(gitignore_path, 'r') as f:
                content = f.read()
            
            if cache_pattern not in content:
                with open(gitignore_path, 'a') as f:
                    f.write(f"\n# Model cache directory (large files)\n{cache_pattern}\n")
                print(f"✅ Added cache directory to .gitignore: {cache_pattern}")
            else:
                print(f"✅ Cache directory already in .gitignore: {cache_pattern}")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not update .gitignore: {str(e)}")
            print(f"⚠️ Please manually add '{cache_pattern}' to your .gitignore file!")

    def _get_cache_path(self, filename):
        """Get the local cache path for a file"""
        return self.cache_dir / f"{self.model_name_or_path.replace('/', '_')}_{filename}"

    def _load_with_cache(self, filename):
        """
        Load a file with local caching to speed up subsequent loads.
        
        Args:
            filename (str): Name of the file to download/load
            
        Returns:
            object: Loaded joblib object
        """
        cache_path = self._get_cache_path(filename)
        
        # Check if file exists in cache
        if cache_path.exists():
            try:
                print(f"Loading {filename} from cache...")
                return joblib.load(cache_path)
            except Exception as e:
                print(f"Cache corrupted for {filename}, re-downloading... ({str(e)})")
                cache_path.unlink()  # Remove corrupted cache file
        
        # Download and cache the file
        return self._download_and_cache(filename, cache_path)

    def _download_and_cache(self, filename, cache_path):
        """Download file from HuggingFace and save to local cache"""
        try:
            file_url = f"https://huggingface.co/{self.model_name_or_path}/resolve/main/{filename}"
            print(f"Downloading {filename} from HuggingFace...")
            
            response = requests.get(file_url, stream=True)  # Use streaming for large files
            response.raise_for_status()
            
            # Save to cache first
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Load from cache
            loaded_object = joblib.load(cache_path)
            print(f"Successfully downloaded and cached {filename}")
            
            return loaded_object
            
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            if cache_path.exists():
                cache_path.unlink()  # Clean up partial download
            raise

    def _download_and_load_pickle(self, filename):
        """
        Fallback method: Download and load a pickle file from HuggingFace without caching.
        (Keeping this for compatibility with your original code)
        
        Args:
            filename (str): Name of the pickle file to download
            
        Returns:
            object: Loaded pickle object
        """
        try:
            file_url = f"https://huggingface.co/{self.model_name_or_path}/resolve/main/{filename}"
            print(f"Downloading {filename} from: {file_url}")
            
            response = requests.get(file_url)
            response.raise_for_status()
            
            # Create a BytesIO object from the response content
            file_buffer = BytesIO(response.content)
            
            # Load using joblib (since you used joblib.dump())
            loaded_object = joblib.load(file_buffer)
            print(f"Successfully loaded {filename}")
            
            return loaded_object
            
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            raise

    @classmethod
    def get_instance(cls, model_name_or_path="habib-ashraf/resume-job-classifier", enable_cache=False):
        """
        Singleton pattern to ensure the classifier is only loaded once.
        """
        if not hasattr(cls, '_instance'):
            cls._instance = cls(model_name_or_path, enable_cache)
        return cls._instance