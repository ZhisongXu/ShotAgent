import json
import sys
from urllib.request import urlopen


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: list_archive_videos.py ARCHIVE_IDENTIFIER")
    with urlopen(f"https://archive.org/metadata/{sys.argv[1]}", timeout=30) as response:
        data = json.load(response)
    for item in data.get("files", []):
        name = item.get("name", "")
        fmt = item.get("format", "")
        if "mp4" in name.lower() or "h.264" in fmt.lower():
            print(name, item.get("size", ""), fmt, sep="\t")


if __name__ == "__main__":
    main()
