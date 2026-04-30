import requests

# Test multiple potential URLs for DocLayNet
urls = [
    "https://huggingface.co/datasets/ds4sd/DocLayNet/resolve/main/COCO/train.json",
    "https://huggingface.co/datasets/ds4sd/DocLayNet/raw/main/COCO/train.json",
    "https://huggingface.co/datasets/ds4sd/DocLayNet-COCO/resolve/main/train.json",
    "https://zenodo.org/record/7626360/files/train.json",
    "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip",
]

for url in urls:
    try:
        r = requests.head(url, allow_redirects=True, timeout=5)
        size = r.headers.get("content-length")
        size_display = f"{size} bytes" if size is not None else "unknown size"
        print(f"[OK] {url[:70]:70} -> {r.status_code} ({size_display})")
        if r.status_code == 200 and size and size.isdigit() and int(size) > 1_000_000:
            print(f"\n  >>> This one looks good! Size: {int(size) / 1048576:.1f} MB")
    except Exception as e:
        print(f"[ER] {url[:70]:70} -> Error: {str(e)[:30]}")
