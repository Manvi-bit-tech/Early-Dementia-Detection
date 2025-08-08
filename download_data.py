import os
import zipfile
import gdown

#  Replace this with your actual Google Drive file ID:
FILE_ID = "1uZ6DifBxwkl1J8y7mnq9QlSkqCQhLCrH"

ZIP_PATH = "data.zip"
EXTRACT_DIR = "data"

def download_from_gdrive(file_id, output_path):
    """
    Download a file from Google Drive using its file ID.
    Uses fuzzy matching to handle different URL formats.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"⬇️  Downloading dataset from Google Drive (ID: {file_id})...")
    gdown.download(url=url, output=output_path, fuzzy=True, quiet=False)

def extract_zip(zip_path, extract_dir):
    """
    Extract the downloaded ZIP into the given directory.
    """
    print(f"📦 Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"✅ Extraction complete. Contents available in '{extract_dir}'.")

if __name__ == "__main__":
    if os.path.exists(EXTRACT_DIR):
        print("✅ Dataset directory already exists. Skipping download.")
    else:
        try:
            download_from_gdrive(FILE_ID, ZIP_PATH)
            if os.path.exists(ZIP_PATH):
                extract_zip(ZIP_PATH, EXTRACT_DIR)
                os.remove(ZIP_PATH)
                print("🗑️ Temporary zip file removed.")
            else:
                print("❌ Download failed — zip file not found.")
        except Exception as e:
            print(f"❗ Error during download/extraction: {e}")
