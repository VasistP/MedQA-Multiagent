# src/utils/download_dataset.py
import os
import requests
import zipfile
import json
from tqdm import tqdm


def download_medqa():
    """
    Download MedQA dataset from the official repository
    """
    # Create data directory
    os.makedirs("./data/medqa", exist_ok=True)

    # MedQA dataset URLs (from the paper's referenced repository)
    base_url = "https://github.com/jind11/MedQA/raw/master/data_clean/questions/US/"
    files = ["train.jsonl", "dev.jsonl", "test.jsonl"]

    print("Downloading MedQA dataset...")
    for file in files:
        url = base_url + file
        output_path = f"./data/medqa/{file}"

        if os.path.exists(output_path):
            print(f"{file} already exists, skipping...")
            continue

        print(f"Downloading {file}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

    print("MedQA dataset downloaded successfully!")


if __name__ == "__main__":
    download_medqa()
