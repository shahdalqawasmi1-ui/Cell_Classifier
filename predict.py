"""Command-line prediction.
Usage:  python predict.py image.tif [more images or folders]  [--csv out.csv]"""
import argparse
import glob
import os
import csv
from cellclf import CellClassifier

EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
ap = argparse.ArgumentParser()
ap.add_argument("inputs", nargs="+")
ap.add_argument("--csv", default=None)
a = ap.parse_args()

paths = []
for x in a.inputs:
    paths += sorted(glob.glob(os.path.join(x, "**", "*"), recursive=True)) if os.path.isdir(x) else [x]
paths = [p for p in paths if p.lower().endswith(EXTS)]

clf = CellClassifier()
rows = []
for p in paths:
    r = clf.predict(p)
    rows.append([os.path.basename(p), r["label"], f"{r['p_differentiated']:.3f}"])
    print(*rows[-1], sep="\t")
if a.csv:
    with open(a.csv, "w", newline="") as f:
        csv.writer(f).writerows([["file", "prediction", "p_differentiated"]] + rows)
