# v2: คลาส other = แดนดิไลออน (TF Flowers) + ดอกหลายชนิดจาก Oxford 102 Flowers
# Oxford แบ่ง "ตามชนิด" (species-disjoint): ชนิดใน test ไม่เคยเห็นตอน train → วัดได้จริงว่าปัดดอกที่ไม่รู้จักเป็น other ได้ไหม
# ต้องรัน 1_split_tfflowers.py ก่อน และมี datasets/oxford102/{raw/jpg, imagelabels.mat}
# ผล: datasets/species_cls_v2/{train,val,test}/{daisy,rose,sunflower,tulip,other}/
import os, glob, random, shutil, json
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "datasets", "tf_flowers_cls")
OX = os.path.join(ROOT, "datasets", "oxford102")
OUT = os.path.join(ROOT, "datasets", "species_cls_v2")
SAME_AS_OURS = {49: "oxeye daisy", 54: "sunflower", 74: "rose"}   # ชนิดซ้ำกับ 4 คลาสของเรา → ไม่เอาเข้า other
PER_CLASS = 8   # รูปต่อชนิดของ Oxford (กันคลาส other ใหญ่เกินคลาสอื่น)

random.seed(0)
if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(V1, OUT, ignore=shutil.ignore_patterns("*.npz", "counts.json"))

labels = loadmat(os.path.join(OX, "imagelabels.mat"))["labels"][0]   # 1..102 ต่อรูป image_00001.jpg ...
by_cls = {}
for i, lb in enumerate(labels, 1):
    by_cls.setdefault(int(lb), []).append(os.path.join(OX, "raw", "jpg", f"image_{i:05d}.jpg"))
classes = sorted(c for c in by_cls if c not in SAME_AS_OURS)
random.shuffle(classes)
n = len(classes)
split_cls = {"train": classes[:round(n * 0.7)], "val": classes[round(n * 0.7):round(n * 0.85)], "test": classes[round(n * 0.85):]}
for split, cs in split_cls.items():
    for c in cs:
        for f in random.sample(by_cls[c], min(PER_CLASS, len(by_cls[c]))):
            shutil.copy(f, os.path.join(OUT, split, "other", f"ox{c:03d}_" + os.path.basename(f)))

counts = {s: {c: len(os.listdir(os.path.join(OUT, s, c))) for c in sorted(os.listdir(os.path.join(OUT, s)))} for s in ("train", "val", "test")}
json.dump({"counts": counts, "oxford_classes": {s: sorted(cs) for s, cs in split_cls.items()},
           "excluded": SAME_AS_OURS}, open(os.path.join(OUT, "counts.json"), "w"), indent=1)
print(json.dumps(counts, indent=1))
assert not set(split_cls["train"]) & set(split_cls["test"])
