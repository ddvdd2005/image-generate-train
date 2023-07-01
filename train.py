from ultralytics.yolo.data.converter import convert_coco

convert_coco(labels_dir='./distorted', cls91to80=False)
