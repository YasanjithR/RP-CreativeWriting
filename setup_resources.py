import nltk
import sys
import subprocess

def download_resources():
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    
    print("Downloading spaCy resources...")
    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    print("All resources downloaded successfully!")

if __name__ == "__main__":
    download_resources() 