"""Core inference: image -> tiles -> DINOv2 embeddings -> logistic regression."""
import os
import numpy as np
import torch
import timm
import joblib
import tifffile
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "cell_classifier.joblib")


class CellClassifier:
    def __init__(self, model_path=MODEL_PATH):
        m = joblib.load(model_path)
        self.clf, self.tile, self.inp, self.classes = m["clf"], m["tile"], m["inp"], m["classes"]
        self.net = timm.create_model(m["backbone"], pretrained=True, num_classes=0, img_size=self.inp).eval()
        cfg = timm.data.resolve_data_config({}, model=self.net)
        self.mean = np.array(cfg["mean"], np.float32)[:, None, None]
        self.std = np.array(cfg["std"], np.float32)[:, None, None]

    @staticmethod
    def load_gray(src):
        """Path, PIL image or numpy array -> 8-bit grayscale PIL image (robust contrast stretch)."""
        if isinstance(src, str):
            a = tifffile.imread(src) if src.lower().endswith((".tif", ".tiff")) else np.array(Image.open(src))
        elif isinstance(src, Image.Image):
            a = np.array(src)
        else:
            a = np.asarray(src)
        a = np.squeeze(a).astype(np.float32)
        if a.ndim == 3:
            if a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
                a = np.moveaxis(a, 0, -1)
            a = a[..., :3].mean(-1)
        lo, hi = np.percentile(a, (0.5, 99.5))
        return Image.fromarray((np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8))

    @torch.no_grad()
    def predict(self, src):
        """Returns dict: label, p_differentiated, tile probabilities grid, grayscale image."""
        im = self.load_gray(src)
        W, H = im.size
        T = min(self.tile, W, H)
        ys = list(range(0, H - T + 1, T))
        xs = list(range(0, W - T + 1, T))
        tiles = []
        for y in ys:
            for x in xs:
                t = np.asarray(im.crop((x, y, x + T, y + T)).resize((self.inp, self.inp), Image.BILINEAR), np.float32) / 255
                tiles.append((np.repeat(t[None], 3, 0) - self.mean) / self.std)
        x = torch.from_numpy(np.stack(tiles).astype(np.float32))
        f = ((self.net(x) + self.net(torch.flip(x, [3]))) / 2).numpy()
        tp = self.clf.predict_proba(f)[:, 1]
        p = float(tp.mean())
        return dict(label=self.classes[int(p >= 0.5)], p_differentiated=p,
                    tile_probs=tp.reshape(len(ys), len(xs)), tile_size=T, image=im)
