#!/usr/bin/env python3
"""Reconstruct submission zip archive from split parts."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
parts = sorted(ROOT.glob("lab21_2A202602474.zip.part*"))
if not parts:
    print("No split parts found.")
    raise SystemExit(1)

out_file = ROOT / "lab21_2A202602474.zip"
with open(out_file, "wb") as out:
    for p in parts:
        out.write(p.read_bytes())

print(f"Reconstructed {out_file.name} ({out_file.stat().st_size / 1024 / 1024:.1f} MB) successfully.")
