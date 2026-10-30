"""Expand the compact repository bundle and verify original frozen split hashes."""

import argparse
import gzip
import json
import shutil
from pathlib import Path
from prepare import file_sha256, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", type=Path, required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.bundle / "manifest.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    for name, expected in original["files"].items():
        source = args.bundle / name
        compressed = args.bundle / (name + ".gz")
        if source.exists():
            shutil.copyfile(source, args.out / name)
        elif compressed.exists():
            with gzip.open(compressed, "rb") as inp, (args.out / name).open(
                "wb"
            ) as out:
                shutil.copyfileobj(inp, out)
        else:
            continue  # Scratch FASTAs/full excluded sequences stay external.
        if file_sha256(args.out / name) != expected:
            raise ValueError("Artifact hash mismatch: " + name)
    required = {"train.csv", "validation.csv", "test.csv", "mmseqs2_clusters.csv"}
    if not required.issubset({p.name for p in args.out.iterdir()}):
        raise ValueError("Incomplete frozen split bundle")
    dump(
        {
            "status": "frozen",
            "protocol": original["protocol"],
            "parent_manifest_sha256": file_sha256(args.bundle / "manifest.json"),
            "note": "Compact export; all included files match the original frozen hashes.",
            "files": {
                p.name: file_sha256(p) for p in args.out.iterdir() if p.is_file()
            },
        },
        args.out / "manifest.json",
    )


if __name__ == "__main__":
    main()
