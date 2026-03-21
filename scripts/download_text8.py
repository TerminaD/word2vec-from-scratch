import os
import shutil
import zipfile
import urllib.request
from pathlib import Path

URL = "http://mattmahoney.net/dc/text8.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "text8.zip")
EXTRACTED_PATH = os.path.join(DATA_DIR, "text8")
TXT_PATH = os.path.join(DATA_DIR, "text8.txt")


def main():
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download ZIP file
    print(f"Downloading {URL} to {ZIP_PATH}")
    with urllib.request.urlopen(URL) as response, open(ZIP_PATH, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    # Unzip ZIP file
    print(f"Extracting {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    # Rename extracted file from `text8` to `text8.txt`
    print(f"Renaming {EXTRACTED_PATH} to {TXT_PATH}")
    if TXT_PATH.exists():
        TXT_PATH.unlink()
    if not EXTRACTED_PATH.exists():
        raise FileNotFoundError(f"Expected extracted file not found: {EXTRACTED_PATH}")
    EXTRACTED_PATH.rename(TXT_PATH)

    # Delete archive
    print(f"Deleting {ZIP_PATH}")
    ZIP_PATH.unlink(missing_ok=True)

    print(f"Finished. Final file: {TXT_PATH}")


if __name__ == "__main__":
    main()
