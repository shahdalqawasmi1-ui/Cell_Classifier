"""Differentiated vs undifferentiated SH-SY5Y classifier.
Frozen pretrained backbone (DINOv2 ViT-S/14) -> tile embeddings -> logistic regression.
Evaluation: leave-one-experiment-date-out cross-validation (no batch leakage)."""
import os, json, numpy as np, pandas as pd, torch, timm, joblib
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

torch.set_num_threads(os.cpu_count())
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'out'); os.makedirs(OUT, exist_ok=True)
BACKBONE = 'vit_small_patch14_dinov2.lvd142m'
TILE, INP = 512, 224

sel = pd.read_csv(os.path.join(ROOT, 'scripts', 'selected_images.csv'))
sel['path'] = [os.path.join(ROOT, 'data', 'images', f'{i:04d}.png') for i in range(len(sel))]
sel['y'] = (sel.first_differentiating_agent == 'RetAc').astype(int)   # 1 = differentiated (RA)
sel['group'] = sel.date_photo.map({1.08: 'Exp 01-Aug', 4.08: 'Exp 04-Aug', 25.07: 'Exp 25-Jul', 28.07: 'Exp 28-Jul'})
sel = sel[[os.path.exists(p) for p in sel.path]].reset_index(drop=True)
print(sel.groupby(['group', 'y']).size().unstack())

model = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=INP).eval()
cfg = timm.data.resolve_data_config({}, model=model)
mean, std = np.array(cfg['mean'])[:, None, None], np.array(cfg['std'])[:, None, None]

def tiles(path):
    im = Image.open(path).convert('L'); W, H = im.size
    out = []
    for y in range(0, H - TILE + 1, TILE):
        for x in range(0, W - TILE + 1, TILE):
            t = np.asarray(im.crop((x, y, x + TILE, y + TILE)).resize((INP, INP), Image.BILINEAR), np.float32) / 255
            out.append((np.repeat(t[None], 3, 0) - mean) / std)
    return np.stack(out).astype(np.float32)

feat_file = f'{OUT}/features.npz'
if os.path.exists(feat_file):
    F = np.load(feat_file, allow_pickle=True); feats = list(F['feats'])
else:
    feats = []
    with torch.no_grad():
        for k, p in enumerate(sel.path):
            x = torch.from_numpy(tiles(p))
            f = (model(x) + model(torch.flip(x, [3]))) / 2      # flip TTA
            feats.append(f.numpy()); print(k, p, f.shape, flush=True)
    np.savez(feat_file, feats=np.array(feats, dtype=object))

def make_clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))

def fit(idx):
    X = np.concatenate([feats[i] for i in idx]); Y = np.concatenate([[sel.y[i]] * len(feats[i]) for i in idx])
    return make_clf().fit(X, Y)

def img_prob(clf, i): return clf.predict_proba(feats[i])[:, 1].mean()

# ---- leave-one-experiment-out CV ----
sel['prob'] = np.nan
fold_rows = []
for g in sorted(sel.group.unique()):
    tr = np.where(sel.group != g)[0]; te = np.where(sel.group == g)[0]
    clf = fit(tr)
    sel.loc[te, 'prob'] = [img_prob(clf, i) for i in te]
    s = sel.loc[te]; pred = (s.prob >= 0.5).astype(int)
    fold_rows.append(dict(test_experiment=g, n=len(s), accuracy=(pred == s.y).mean(),
                          sensitivity=(pred[s.y == 1] == 1).mean(), specificity=(pred[s.y == 0] == 0).mean(),
                          auc=roc_auc_score(s.y, s.prob)))
folds = pd.DataFrame(fold_rows); print(folds.round(3))

y, p = sel.y.values, sel.prob.values; pred = (p >= 0.5).astype(int)
rng = np.random.default_rng(0)
def boot(fn):
    v = []
    for _ in range(2000):
        b = rng.integers(0, len(y), len(y))
        if len(set(y[b])) == 2: v.append(fn(y[b], p[b]))
    return np.percentile(v, [2.5, 97.5])
acc = (pred == y).mean(); auc = roc_auc_score(y, p)
tn, fp, fn_, tp = confusion_matrix(y, pred).ravel()
summary = dict(n_images=int(len(y)), n_differentiated=int(y.sum()), n_undifferentiated=int((1 - y).sum()),
               accuracy=acc, accuracy_CI=boot(lambda a, b: ((b >= .5) == a).mean()).tolist(),
               sensitivity=tp / (tp + fn_), specificity=tn / (tn + fp), auc=auc,
               auc_CI=boot(roc_auc_score).tolist(), confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn_), tp=int(tp)))
print(json.dumps(summary, indent=1, default=float))
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
folds.to_csv(f'{OUT}/per_experiment_results.csv', index=False)
sel[['image_name', 'group', 'first_differentiating_agent', 'first_agent_concentration(micromolar)',
     'FBS_percentage', 'phorbol_ester', 'y', 'prob']].to_csv(f'{OUT}/per_image_predictions.csv', index=False)

# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
cm = np.array([[tn, fp], [fn_, tp]])
ax[0].imshow(cm, cmap='Blues')
for (r, c), v in np.ndenumerate(cm): ax[0].text(c, r, v, ha='center', va='center', fontsize=16, color='white' if v > cm.max() / 2 else 'black')
ax[0].set_xticks([0, 1], ['Undiff.', 'Differentiated']); ax[0].set_yticks([0, 1], ['Undiff.', 'Differentiated'])
ax[0].set_xlabel('Predicted'); ax[0].set_ylabel('True'); ax[0].set_title(f'Confusion matrix (acc {acc:.1%})')
fpr, tpr, _ = roc_curve(y, p)
ax[1].plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}'); ax[1].plot([0, 1], [0, 1], '--', c='grey')
ax[1].set_xlabel('1 − specificity'); ax[1].set_ylabel('Sensitivity'); ax[1].set_title('ROC (leave-one-experiment-out)'); ax[1].legend()
plt.tight_layout(); plt.savefig(f'{OUT}/results.png', dpi=150)

fig, ax = plt.subplots(2, 3, figsize=(12, 7))
for r, cls in enumerate([0, 1]):
    for c, i in enumerate(sel[sel.y == cls].sample(3, random_state=1).index):
        im = Image.open(sel.path[i]); ax[r, c].imshow(im.crop((800, 500, 1824, 1268)), cmap='gray'); ax[r, c].axis('off')
        ax[r, c].set_title(f"{'RA-differentiated' if cls else 'Undifferentiated'} | P(diff)={sel.prob[i]:.2f}", fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/examples.png', dpi=120)

# ---- final model on all data ----
final = fit(np.arange(len(sel)))
joblib.dump(dict(clf=final, backbone=BACKBONE, tile=TILE, inp=INP, classes=['Undifferentiated', 'Differentiated']),
            f'{OUT}/cell_classifier.joblib')
print('saved')
