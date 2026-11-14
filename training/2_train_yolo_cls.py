# แบบ A: เทรน YOLO11n-cls บอกชนิดดอก (daisy/rose/sunflower/tulip/other) จาก datasets/tf_flowers_cls
# ผล: runs/classify/species_yolo11n_cls/weights/best.pt
import os
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":   # Windows: worker ของ dataloader import ไฟล์นี้ซ้ำ ต้องกันไว้
    os.chdir(ROOT)
    model = YOLO("yolo11n-cls.pt")   # เริ่มจากน้ำหนัก ImageNet
    model.train(data=os.path.join(ROOT, "datasets", "tf_flowers_cls"), epochs=30, imgsz=224, batch=32,
                project=os.path.join(ROOT, "runs", "classify"), name="species_yolo11n_cls", exist_ok=True,
                seed=0, workers=2, patience=10)
