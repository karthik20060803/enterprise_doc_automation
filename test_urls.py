import requests

# Test multiple potential URLs for DocLayNet
urls = [
    "https://huggingface.co/datasets/ds4sd/DocLayNet/resolve/main/COCO/train.json",
    "https://huggingface.co/datasets/ds4sd/DocLayNet/raw/main/COCO/train.json",
    "https://huggingface.co/datasets/ds4sd/DocLayNet-COCO/resolve/main/train.json",
    "https://zenodo.org/record/7626360/files/train.json",
]

for url in urls:
    try:
        r = requests.head(url, allow_redirects=True, timeout=5)
        size = r.headers.get('content-length', 'unknown')
        print(f"✓ {url[:70]:70} → {r.status_code} ({size} bytes)")
        if r.status_code == 200 and int(r.headers.get('content-length', 0)) > 1000000:
            print(f"\n  >>> This one looks good! Size: {int(r.headers.get('content-length', 0))/1048576:.1f} MB")
    except Exception as e:
        print(f"✗ {url[:70]:70} → Error: {str(e)[:30]}")
