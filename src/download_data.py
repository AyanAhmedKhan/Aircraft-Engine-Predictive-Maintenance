import os
import urllib.request
import zipfile
import io
import ssl

# Bypass SSL verification if needed (sometimes helps with corporate networks or specific setups)
ssl._create_default_https_context = ssl._create_unverified_context

DATA_URL = "https://ti.arc.nasa.gov/c/6/"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "CMAPSSData.zip")

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Downloading data from {DATA_URL}...")
    try:
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Download complete.")
        
        print("Extracting data...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(DATA_DIR)
        print("Data extracted successfully.")
        
        # Verify files
        files = os.listdir(DATA_DIR)
        print("Files in data directory:", files)
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
