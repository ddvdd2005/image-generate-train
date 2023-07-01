from ultralytics.yolo.data.converter import convert_coco
import shutil
import json
from ultralytics import YOLO

convert_coco(labels_dir='./distorted', cls91to80=False)
shutil.move("distorted/val", "./yolo_labels/images/val")
shutil.move("./distorted/train", "./yolo_labels/images/train")
with open("./distorted/train.json", "r") as f:
    coco_json = json.load(f)
yaml_content = """path: ./yolo_labels
train: images/train
val: images/val
classes:
"""
for subclass in coco_json["categories"]:
    yaml_content += f"  {subclass['id']}: {subclass['name']}\n"
with open("./yolo_labels/train.yaml", "w") as f:
    f.write(yaml_content)

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='./yolo_labels/train.yaml', epochs=100, imgsz=640)
