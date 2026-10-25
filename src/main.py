"""Compatibility wrapper. Prefer `python infer.py photo.jpg` on test day."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent.parent
infer_path = ROOT / "infer.py"
sys.argv[0] = str(infer_path)
runpy.run_path(str(infer_path), run_name="__main__")
