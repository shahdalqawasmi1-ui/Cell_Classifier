import io, os, sys, numpy as np, pandas as pd, tifffile
from PIL import Image
from remotezip import RemoteZip
from concurrent.futures import ThreadPoolExecutor

URL = 'https://zenodo.org/api/records/19634700/files/PhaseContrastNeuriteAnalyzer_dataset_v1.0.0.zip/content'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'images')
sel = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'selected_images.csv'))
os.makedirs(OUT, exist_ok=True)

with RemoteZip(URL) as z:
    name_map = {n.replace('\\', '/'): n for n in z.namelist()}

def to8(a):
    a = np.squeeze(a).astype(np.float32)
    if a.ndim == 3:
        if a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4): a = np.moveaxis(a, 0, -1)
        a = a[..., :3].mean(-1)
    lo, hi = np.percentile(a, (0.5, 99.5))
    return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)

def job(row):
    i, r = row
    out = f"{OUT}/{i:04d}.png"
    if os.path.exists(out): return out, 'skip'
    key = name_map[r.zenodo_raw_relative_path]
    with RemoteZip(URL) as z:
        raw = z.read(key)
    if key.lower().endswith('.tif'):
        a = tifffile.imread(io.BytesIO(raw))
    else:
        a = np.array(Image.open(io.BytesIO(raw)))
    img = Image.fromarray(to8(a))
    img.save(out)
    return out, img.size

with ThreadPoolExecutor(6) as ex:
    for k, (out, info) in enumerate(ex.map(job, sel.iterrows()), 1):
        print(k, out, info, flush=True)
print('DONE')
