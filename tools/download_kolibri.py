#!/usr/bin/env python3
"""Download Kolibri-1 Q4_K_M with parallel connections and no bandwidth cap.

    python tools/download_kolibri.py --dry-run
    python tools/download_kolibri.py --connections 32 --output-dir /data/Kolibri-1

Interrupted downloads resume on the next run. Downloading does not add Kolibri
support to the Strata engine.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

REPO = "bartowski/Kolibri-1-GGUF"
REVISION = "ffd6001f205725f0c4af4ff93e33c04f0f8ad9f9"
FILENAME = "Kolibri-1-Q4_K_M.gguf"
SIZE = 49_759_219_584
SHA256 = "2f2670a78b5c21bef65ef286791285d8e0ad79aef5f0904a3784cb7eeadd5f3e"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT.parent / "Strata-data" / "models" / "Kolibri-1",
                        help="destination folder (default: ../Strata-data/models/Kolibri-1 beside this checkout)")
    parser.add_argument("--connections", type=int, default=setup.dl_connections(), metavar="N",
                        help="parallel connections, 1–32 (default: STRATA_DL_CONNECTIONS or 16)")
    parser.add_argument("--dry-run", action="store_true", help="show the download without writing files or using the network")
    args = parser.parse_args(argv)
    if not 1 <= args.connections <= 32:
        parser.error("--connections must be between 1 and 32")

    destination = args.output_dir.expanduser().resolve() / FILENAME
    url = f"{setup.hf_endpoint()}/{REPO}/resolve/{REVISION}/{FILENAME}"
    print(f"Source: {url}\nDestination: {destination}\nSize: {SIZE / 1e9:.2f} GB ({SIZE:,} bytes)")
    print(f"Connections: {args.connections}; no bandwidth cap\nSHA-256: {SHA256}", flush=True)
    if args.dry_run:
        return 0

    # Reuse setup's resumable downloader without changing its model catalog on disk.
    setup.REPO_PINS[REPO] = {FILENAME: (SIZE, SHA256)}
    os.environ["STRATA_DL_CONNECTIONS"] = str(args.connections)
    os.environ["STRATA_SOURCE"] = "huggingface"
    os.environ["STRATA_ALLOW_UNPINNED"] = "0"
    setup.download(url, destination, "Kolibri-1 Q4_K_M")
    setup.verify_sha256(destination, SIZE, SHA256)
    print(f"Ready: {destination}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(setup.acquisition.main_guard(main))
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run the same command to resume.", file=sys.stderr)
        sys.exit(130)
