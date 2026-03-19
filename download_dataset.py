import requests
from tqdm import tqdm
import os

# DocLayNet train annotations URL (correct repository: pierreguillou/DocLayNet-base)
url = "https://huggingface.co/datasets/pierreguillou/DocLayNet-base/resolve/main/DocLayNet/COCO/train.json"
output_file = "dataset/train.json"

# Ensure dataset directory exists
os.makedirs("dataset", exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}

print("Starting download of DocLayNet train annotations...")

response = requests.get(url, headers=headers, stream=True)

total_size = int(response.headers.get('content-length', 0))

with open(output_file, "wb") as file:
    with tqdm(total=total_size, unit='B', unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

print("Download finished.")