"""Web interface: upload phase-contrast images -> differentiated / undifferentiated + tile heatmap.
Run locally:  python app.py   (then open http://127.0.0.1:7860)"""
import glob
import os
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

from cellclf import CellClassifier

clf = CellClassifier()
HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = sorted(glob.glob(os.path.join(HERE, "examples", "*.jpg")))


def overlay(res, alpha=0.35):
    im = res["image"].convert("RGB")
    T, grid = res["tile_size"], res["tile_probs"]
    heat = np.zeros((im.height, im.width), np.float32)
    mask = np.zeros_like(heat)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            heat[r * T:(r + 1) * T, c * T:(c + 1) * T] = grid[r, c]
            mask[r * T:(r + 1) * T, c * T:(c + 1) * T] = 1
    col = (cm.RdBu_r(heat)[..., :3] * 255).astype(np.float32)
    base = np.asarray(im, np.float32)
    a = alpha * mask[..., None]
    out = Image.fromarray((base * (1 - a) + col * a).astype(np.uint8))
    out.thumbnail((1400, 1400))
    return out


def classify_one(path):
    if path is None:
        return None, None, ""
    res = clf.predict(path)
    p = res["p_differentiated"]
    conf = max(p, 1 - p)
    note = ""
    if conf < 0.65:
        note = "\n\n⚠️ Low confidence: the image looks intermediate (e.g. partial or early differentiation)."
    md = (f"## {res['label']}\n"
          f"**P(differentiated) = {p:.2f}** · confidence {conf:.0%}{note}\n\n"
          f"Heatmap: red tiles look differentiated, blue tiles look undifferentiated.")
    return overlay(res), {"Differentiated": p, "Undifferentiated": 1 - p}, md


def classify_batch(files):
    if not files:
        return None, None
    rows = []
    for f in files:
        path = f if isinstance(f, str) else f.name
        r = clf.predict(path)
        rows.append({"file": os.path.basename(path), "prediction": r["label"],
                     "P(differentiated)": round(r["p_differentiated"], 3)})
    df = pd.DataFrame(rows)
    out = os.path.join("/tmp", "predictions.csv")
    df.to_csv(out, index=False)
    return df, out


ABOUT = """
### About this model
- **Task:** differentiated vs undifferentiated cells from **phase-contrast** images.
- **Training data:** 229 open images of SH-SY5Y cells (retinoic-acid-differentiated vs untreated controls), 4 independent experiments
  ([Yordanov & Hristiva, Zenodo 19634700, CC-BY 4.0](https://zenodo.org/records/19634700)).
- **Method:** each image is split into 512-px tiles → frozen DINOv2 ViT-S/14 embeddings → logistic regression. Image score = mean of its tiles.
- **Validation (leave-one-experiment-out):** accuracy **87.8%** (95% CI 83–92%), AUC **0.92**, sensitivity 91%, specificity 84%.
- **Limits:** pilot model trained on one cell line, one microscope, ~10× phase contrast. Retrain on your own lab images
  (e.g. MSC vs fibroblast) before research use. Not for clinical use.
"""

with gr.Blocks(title="Cell Differentiation Classifier") as demo:
    gr.Markdown("# 🔬 Cell Differentiation Classifier\nUpload a phase-contrast micrograph (TIFF, PNG, JPG).")
    with gr.Tab("Single image"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="filepath", label="Micrograph", height=380)
                btn = gr.Button("Classify", variant="primary")
                if EXAMPLES:
                    gr.Examples(EXAMPLES, inputs=inp, label="Examples (click one)")
            with gr.Column():
                txt = gr.Markdown()
                lab = gr.Label(num_top_classes=2, label="Probabilities")
                heat = gr.Image(label="Tile heatmap", height=380)
        btn.click(classify_one, inp, [heat, lab, txt])
    with gr.Tab("Batch (many images)"):
        files = gr.File(file_count="multiple", label="Upload images")
        bbtn = gr.Button("Classify all", variant="primary")
        table = gr.Dataframe(label="Results")
        csv = gr.File(label="Download CSV")
        bbtn.click(classify_batch, files, [table, csv])
    with gr.Tab("About"):
        gr.Markdown(ABOUT)
        gr.Image(os.path.join(HERE, "results", "results.png"), label="Validation results", show_label=True)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
