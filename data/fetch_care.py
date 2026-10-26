"""Fetch only Wind Farm A from the CARE to Compare zip on Zenodo (HTTP range requests, no 5.5 GB download).

Pinned: Zenodo record 15846963 (version 6, DOI 10.5281/zenodo.15846963) - the latest version of the
concept record 10958774. Version 5 (record 14958989, used by the old USER_MANUAL results) can be fetched
with --v5 into data/CARE_To_Compare_v5/ to reproduce the legacy numbers.
Usage:  python data/fetch_care.py            -> data/CARE_To_Compare/Wind Farm A/...
        python data/fetch_care.py --v5       -> data/CARE_To_Compare_v5/Wind Farm A/...
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import requests

RECORD = "15846963"
DOI = "10.5281/zenodo.15846963"
URL = f"https://zenodo.org/api/records/{RECORD}/files/CARE_To_Compare.zip/content"
OUT = Path(__file__).resolve().parent / "CARE_To_Compare"


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url):
        self.s = requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
        self.url = r.url
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for _ in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=300)
                r.raise_for_status()
                break
            except requests.RequestException:
                continue
        else:
            raise IOError("range request failed")
        self.pos += len(r.content)
        return r.content

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def main(farm="Wind Farm A"):
    hf = HttpFile(URL)
    zf = zipfile.ZipFile(io.BufferedReader(hf, buffer_size=8 << 20))
    names = [n for n in zf.namelist() if f"{farm}/" in n or n.lower().endswith(("readme.md", "readme.txt", ".pdf"))]
    print(f"{len(names)} members to extract")
    for n in names:
        if n.endswith("/"):
            continue
        tail = n.split(f"{farm}/", 1)[1] if f"{farm}/" in n else Path(n).name
        dest = OUT / (farm if f"{farm}/" in n else "") / tail
        if dest.exists() and dest.stat().st_size == zf.getinfo(n).file_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(n) as src, open(dest, "wb") as dst:
            while chunk := src.read(8 << 20):
                dst.write(chunk)
        print("ok", dest.relative_to(OUT), flush=True)
    (OUT / "VERSION.json").write_text(json.dumps({"record": RECORD, "doi": DOI, "version": 6, "farm": farm}, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--v5" in args:
        args.remove("--v5")
        RECORD, DOI = "14958989", "10.5281/zenodo.14958989"
        URL = f"https://zenodo.org/api/records/{RECORD}/files/CARE_To_Compare.zip/content"
        OUT = Path(__file__).resolve().parent / "CARE_To_Compare_v5"
    main(*args)
