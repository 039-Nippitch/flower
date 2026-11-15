# แบบ A: เทรน YOLO11n-cls บอกชนิดดอก (daisy/rose/sunflower/tulip/other)
# ใช้: python 2_train_yolo_cls.py      (v1: datasets/tf_flowers_cls → runs/classify/species_yolo11n_cls)
#     python 2_train_yolo_cls.py v2   (v2: datasets/species_cls_v2 → runs/classify/species_yolo11n_cls_v2)
import os, sys
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = sys.argv[1:] == ["v2"]

if __name__ == "__main__":   # Windows: worker ของ dataloader import ไฟล์นี้ซ้ำ ต้องกันไว้
    os.chdir(ROOT)
    model = YOLO("yolo11n-cls.pt")   # เริ่มจากน้ำหนัก ImageNet
    model.train(data=os.path.join(ROOT, "datasets", "species_cls_v2" if V2 else "tf_flowers_cls"), epochs=30, imgsz=224,
                batch=32, project=os.path.join(ROOT, "runs", "classify"),
                name="species_yolo11n_cls_v2" if V2 else "species_yolo11n_cls", exist_ok=True, seed=0, workers=2, patience=10)
