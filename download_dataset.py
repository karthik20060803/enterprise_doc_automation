import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URLS = [
    "https://huggingface.co/datasets/ds4sd/DocLayNet/resolve/main/COCO/train.json",
    "https://huggingface.co/datasets/ds4sd/DocLayNet/raw/main/COCO/train.json",
    "https://zenodo.org/records/7626360/files/train.json",
]


def download_file(url: str, output_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise urllib.error.HTTPError(
                url=url,
                code=response.status,
                msg=f"Unexpected HTTP status: {response.status}",
                hdrs=response.headers,
                fp=None,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DocLayNet COCO train annotations (train.json)."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="Optional direct URL for train.json. If omitted, script tries known mirrors.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("dataset") / "train.json"),
        help="Output path for downloaded JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        print(f"File already exists: {output_path}")
        print("Use --force to re-download.")
        return 0

    urls = [args.url] if args.url else DEFAULT_URLS
    errors = []

    for index, url in enumerate(urls, start=1):
        if not url:
            continue
        print(f"[{index}/{len(urls)}] Trying: {url}")
        try:
            download_file(url, output_path)
            file_size_kb = output_path.stat().st_size / 1024
            print(f"Download complete: {output_path} ({file_size_kb:.1f} KB)")
            return 0
        except Exception as exc:
            errors.append((url, str(exc)))
            print(f"Failed: {exc}")

    print("\nAll download attempts failed.")
    print("Pass a working URL manually using --url.")
    for url, err in errors:
        print(f"- {url}\n  -> {err}")
    return 1


if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    sys.exit(main())
