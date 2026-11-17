# แบบ B: BioCLIP 2 (ไม่เทรนตัวมันเอง) ดึง feature → เทรน linear head (logistic regression) บอกชนิด
# แล้ววัดผลบน test เทียบ 3 แบบ: BioCLIP zero-shot (แบบที่เดโมใช้อยู่) | BioCLIP + head ที่เราเทรน | YOLO11n-cls ที่เราเทรน
# ผล: runs/classify/species_bioclip_head{_v2}.pt, results/species_eval{_v2}/{metrics.json, cm_*.png, compare.png}
# ใช้: python 3_bioclip_head_and_eval.py        (v1: other = แดนดิไลออน)
#     python 3_bioclip_head_and_eval.py v2     (v2: other = แดนดิไลออน + Oxford 102, ต้องรัน 1b ก่อน)
import os, glob, json, ast
import numpy as np
import torch, open_clip
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
V2 = sys.argv[1:] == ["v2"]
SUF = "_v2" if V2 else ""
DATA = os.path.join(ROOT, "datasets", "species_cls_v2" if V2 else "tf_flowers_cls")
OUT = os.path.join(ROOT, "results", "species_eval" + SUF)
os.makedirs(OUT, exist_ok=True)
CLASSES = ["daisy", "rose", "sunflower", "tulip", "other"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def files(split):
    return [(f, i) for i, c in enumerate(CLASSES) for f in sorted(glob.glob(os.path.join(DATA, split, c, "*.jpg")))]


# ---------- BioCLIP 2 features (เก็บ cache ไว้ รันซ้ำไม่ต้องคำนวณใหม่) ----------
bio, _, pre = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
bio = bio.to(dev).eval().half() if dev == "cuda" else bio.eval()
tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")


@torch.inference_mode()
def feats(split):
    cache = os.path.join(DATA, f"bioclip2_{split}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["x"], z["y"]
    fs = files(split)
    xs = []
    for k in range(0, len(fs), 32):
        b = torch.stack([pre(Image.open(f).convert("RGB")) for f, _ in fs[k:k + 32]]).to(dev).to(next(bio.parameters()).dtype)
        e = bio.encode_image(b).float()
        xs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    x, y = np.concatenate(xs), np.array([c for _, c in fs])
    np.savez(cache, x=x, y=y)
    return x, y


Xtr, ytr = feats("train"); Xva, yva = feats("val"); Xte, yte = feats("test")
print("features", Xtr.shape, Xva.shape, Xte.shape, flush=True)

# ---------- 1) zero-shot: รายชื่อ 55 ชนิด + prompt เดียวกับโน้ตบุ๊ก → ชนิดที่ไม่ใช่ 4 ชนิด / ความมั่นใจ < 0.35 = other ----------
nb = json.load(open(os.path.join(ROOT, "old_versions", "flower_counter_demo_4ชนิด.ipynb"), encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))   # คำสั่ง %pip / ! ไม่ใช่ Python
ns = {}
for node in ast.parse(src).body:   # ดึงแค่ flowers / templates / SPECIES_THR จากโน้ตบุ๊ก ไม่ต้องรันทั้งไฟล์
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in ("flowers", "templates", "SPECIES_THR"):
        exec(compile(ast.Module([node], []), "nb", "exec"), ns)
names = list(ns["flowers"])
with torch.inference_mode():
    T = []
    for n in names:
        sci, common = ns["flowers"][n]
        t = bio.encode_text(tok([tp.format(sci=sci, common=common) for tp in ns["templates"]]).to(dev)).float()
        t = (t / t.norm(dim=-1, keepdim=True)).mean(0)
        T.append(t / t.norm())
    T = torch.stack(T).cpu().numpy()
    scale = bio.logit_scale.exp().item()
p = torch.tensor(scale * Xte @ T.T).softmax(-1).numpy()
zs = np.array([CLASSES.index(names[i]) if names[i] in CLASSES[:4] and p[k, i] >= ns["SPECIES_THR"] else 4
               for k, i in enumerate(p.argmax(1))])

# ---------- 2) linear head บน feature ของ BioCLIP 2 (เลือก weight decay ด้วย val) ----------
def train_head(wd):
    torch.manual_seed(0)
    W = torch.nn.Linear(Xtr.shape[1], len(CLASSES))
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=wd)
    X, y = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(300):   # full-batch
        opt.zero_grad()
        torch.nn.functional.cross_entropy(W(X * scale ** 0.5), y).backward()
        opt.step()
    return W


def pred(W, X):
    with torch.no_grad():
        return W(torch.tensor(X) * scale ** 0.5).argmax(1).numpy()


best = max(((wd, train_head(wd)) for wd in [0, 1e-4, 1e-3, 1e-2, 1e-1]), key=lambda t: (pred(t[1], Xva) == yva).mean())
head = best[1]
torch.save({"state_dict": head.state_dict(), "classes": CLASSES, "scale": scale ** 0.5, "backbone": "hf-hub:imageomics/bioclip-2"},
           os.path.join(ROOT, "runs", "classify", f"species_bioclip_head{SUF}.pt"))
lp = pred(head, Xte)
print("head wd", best[0], "val acc", (pred(head, Xva) == yva).mean(), flush=True)

# ---------- 2b) head + ด่านกันชนิดอื่น (แบบที่ใส่ในเดโม): zero-shot มั่นใจว่าเป็นชนิดนอก 4 ชนิด → other ----------
# other ใน train มีแค่แดนดิไลออน ด่านนี้กันดอกชนิดอื่นที่ head ไม่เคยเห็น (ลิลลี่ กล้วยไม้ ฯลฯ)
GUARD_THR = 0.9
with torch.no_grad():
    hp = (head(torch.tensor(Xte) * scale ** 0.5)).softmax(-1).numpy()
zs_top, zs_p = p.argmax(1), p.max(1)
guard = np.array([4 if names[zs_top[k]] not in CLASSES[:4] and zs_p[k] >= GUARD_THR else hp[k].argmax() for k in range(len(yte))])
for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    g = np.array([4 if names[zs_top[k]] not in CLASSES[:4] and zs_p[k] >= t else hp[k].argmax() for k in range(len(yte))])
    print(f"guard thr {t}: test acc {(g == yte).mean():.3f}", flush=True)

# ---------- 3) YOLO11n-cls ที่เทรนไว้ (2_train_yolo_cls.py) ----------
yolo_w = os.path.join(ROOT, "runs", "classify", "species_yolo11n_cls" + SUF, "weights", "best.pt")
res = {"BioCLIP 2 zero-shot": zs, "BioCLIP 2 + head (เราเทรน)": lp, "BioCLIP 2 + head + guard (ใช้ในเดโม)": guard}
if os.path.exists(yolo_w):
    from ultralytics import YOLO
    ym = YOLO(yolo_w)
    order = [ym.names[i] for i in range(len(ym.names))]
    yp = []
    for f, _ in files("test"):
        yp.append(CLASSES.index(order[ym.predict(f, imgsz=224, verbose=False)[0].probs.top1]))
    res["YOLO11n-cls (เราเทรน)"] = np.array(yp)

# ---------- วัดผล + รูป ----------
plt.rcParams["font.family"] = ["Tahoma", "DejaVu Sans"]   # Tahoma มีอักษรไทยบน Windows
metrics = {}
for k, yp in res.items():
    cm = np.zeros((5, 5), int)
    for t, q in zip(yte, yp):
        cm[t, q] += 1
    prec = cm.diagonal() / np.maximum(1, cm.sum(0)); rec = cm.diagonal() / np.maximum(1, cm.sum(1))
    f1 = 2 * prec * rec / np.maximum(1e-9, prec + rec)
    metrics[k] = {"accuracy": round(float((yp == yte).mean()), 4), "macro_f1": round(float(f1.mean()), 4),
                  "per_class_f1": {c: round(float(v), 3) for c, v in zip(CLASSES, f1)}, "confusion_matrix": cm.tolist()}
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Greens")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{k}\nacc {metrics[k]['accuracy']:.1%} | macro-F1 {metrics[k]['macro_f1']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cm_" + k.split(" (")[0].replace(" ", "_").replace("+", "plus") + ".png"), dpi=150)
    plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 3.2))
ks = list(metrics)
ax.barh(ks, [metrics[k]["accuracy"] for k in ks], color="#5b8c5a")
for i, k in enumerate(ks):
    ax.text(metrics[k]["accuracy"] + 0.005, i, f"{metrics[k]['accuracy']:.1%}", va="center")
ax.set_xlim(0, 1.08); ax.set_xlabel(f"Test accuracy ({'TF Flowers + Oxford 102' if V2 else 'TF Flowers'} test, n={len(yte)})")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare.png"), dpi=150); plt.close(fig)
# ดอกชนิดที่ไม่เคยเห็นตอนเทรน (Oxford ชุด test แบ่งตามชนิด) ถูกปัดเป็น other กี่ %
ox = np.array([os.path.basename(f).startswith("ox") for f, _ in files("test")])
if ox.any():
    for k, yp in res.items():
        metrics[k]["unseen_species_to_other"] = round(float((yp[ox] == 4).mean()), 4)
        print(f"{k:32s} unseen species → other {metrics[k]['unseen_species_to_other']:.1%} (n={ox.sum()})")
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, m in metrics.items():
    print(f"{k:32s} acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  {m['per_class_f1']}")
