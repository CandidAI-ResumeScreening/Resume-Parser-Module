import requests
import joblib
from io import BytesIO

def test_file_download(model_path, filename):
    """Test downloading and loading a single file from HuggingFace"""
    try:
        url = f"https://huggingface.co/{model_path}/resolve/main/{filename}"
        print(f"Testing download: {url}")
        
        response = requests.get(url)
        print(f"Response status: {response.status_code}")
        print(f"Content length: {len(response.content)} bytes")
        print(f"Content type: {response.headers.get('content-type', 'unknown')}")
        
        # Check first few bytes
        first_bytes = response.content[:20]
        print(f"First 20 bytes: {first_bytes}")
        
        if response.status_code == 200:
            # Try loading with joblib
            file_buffer = BytesIO(response.content)
            loaded_object = joblib.load(file_buffer)
            print(f"✅ Successfully loaded {filename}")
            print(f"Object type: {type(loaded_object)}")
            return True
        else:
            print(f"❌ Failed to download {filename}")
            return False
            
    except Exception as e:
        print(f"❌ Error with {filename}: {str(e)}")
        return False

if __name__ == "__main__":
    model_path = "habib-ashraf/resume-job-classifier"
    files = ["model_exp1.pkl", "tfidf_exp1.pkl", "encoder_exp1.pkl"]
    
    print("🔍 DEBUGGING HUGGINGFACE FILE DOWNLOADS")
    print("="*60)
    
    for filename in files:
        print(f"\n📁 Testing {filename}:")
        print("-" * 40)
        success = test_file_download(model_path, filename)
        print("-" * 40)
        
    print("\n" + "="*60)
    print("Debug complete!")