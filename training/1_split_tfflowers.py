# แบ่งรูป TF Flowers เป็น train/val/test (70/15/15 ต่อคลาส, seed คงที่) สำหรับเทรนตัวบอกชนิด 4 ชนิด + other
# ผล: datasets/tf_flowers_cls/{train,val,test}/{daisy,rose,sunflower,tulip,other}/*.jpg  (โครงแบบที่ YOLO-cls ใช้)
import os, glob, random, shutil, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "datasets", "extra_data", "tf_flowers", "flower_photos")
OUT = os.path.join(ROOT, "datasets", "tf_flowers_cls")
CLASSES = {"daisy": "daisy", "roses": "rose", "sunflowers": "sunflower", "tulips": "tulip", "dandelion": "other"}

random.seed(0)
if os.path.exists(OUT):
    shutil.rmtree(OUT)
counts = {}
for src, cls in CLASSES.items():
    files = sorted(glob.glob(os.path.join(SRC, src, "*.jpg")))
    random.shuffle(files)
    n = len(files)
    parts = {"train": files[:int(n * 0.7)], "val": files[int(n * 0.7):int(n * 0.85)], "test": files[int(n * 0.85):]}
    for split, fs in parts.items():
        d = os.path.join(OUT, split, cls)
        os.makedirs(d)
        for f in fs:
            shutil.copy(f, d)
        counts.setdefault(split, {})[cls] = len(fs)
json.dump(counts, open(os.path.join(OUT, "counts.json"), "w"), indent=1)
print(json.dumps(counts, indent=1))
assert sum(sum(v.values()) for v in counts.values()) == sum(len(glob.glob(os.path.join(SRC, s, "*.jpg"))) for s in CLASSES)
