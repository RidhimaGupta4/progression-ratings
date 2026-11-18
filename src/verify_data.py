"""Verify the downloaded StatsBomb files against data_manifest.tsv (git blob SHA-1 at the pinned commit).
Usage: python src/verify_data.py <statsbomb-open-data checkout>"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def blob_sha1(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main(dest):
    dest = Path(dest)
    bad, n = [], 0
    for line in (ROOT / "data_manifest.tsv").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        rel, sha = line.split("\t")
        n += 1
        f = dest / rel
        if not f.exists() or blob_sha1(f) != sha:
            bad.append(rel)
    if bad:
        sys.exit(f"{len(bad)} of {n} files missing or changed, e.g. {bad[:5]}")
    print(f"verified {n} files against data_manifest.tsv")


if __name__ == "__main__":
    main(sys.argv[1])
