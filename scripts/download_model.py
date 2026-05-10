#!/usr/bin/env python3
"""
Download a model snapshot from Hugging Face to a local `models/` directory.

Defaults to `google/vit-base-patch16-224` and uses the provided token by default.

Usage:
  python scripts/download_model.py
  python scripts/download_model.py --token YOUR_TOKEN --out models/my_vit

Note: Prefer setting `HUGGINGFACE_TOKEN` env var instead of embedding tokens in code.
"""
import os
import sys
import argparse
from pathlib import Path


DEFAULT_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")


def main():
    parser = argparse.ArgumentParser(description="Download HF model snapshot to local directory")
    parser.add_argument(
        "--repo",
        default="google/vit-base-patch16-224",
        help="Hugging Face repository id (e.g. google/vit-base-patch16-224)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (or set HUGGINGFACE_TOKEN env var)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output local directory (default: models/<repo> with / replaced by _)",
    )
    parser.add_argument("--revision", default=None, help="Optional revision/branch/tag to download")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("Missing dependency: install with `pip install huggingface_hub`", file=sys.stderr)
        sys.exit(2)

    repo_id = args.repo
    token = args.token or DEFAULT_TOKEN

    if not token:
        print(
            "Error: no Hugging Face token provided. Set HUGGINGFACE_TOKEN or pass --token",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.out:
        out_dir = Path(args.out)
    else:
        safe_name = repo_id.replace("/", "_")
        out_dir = Path("models") / safe_name

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{repo_id}' to '{out_dir}'")

    try:
        # Prefer new `local_dir` parameter (downloads directly into the given folder)
        path = snapshot_download(repo_id=repo_id, token=token, local_dir=str(out_dir), revision=args.revision)
    except TypeError:
        # Fallback for older huggingface_hub versions that don't support local_dir
        path = snapshot_download(repo_id=repo_id, token=token, cache_dir=str(out_dir), revision=args.revision)

    print("Done. Snapshot path:", path)

    # Show a short listing to confirm
    try:
        entries = list(out_dir.iterdir())
        if entries:
            print("Top-level files:")
            for e in entries[:20]:
                print(" -", e.name)
        else:
            print("Warning: output directory is empty")
    except Exception:
        pass


if __name__ == "__main__":
    main()
