from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("./runs/detect/train6/weights/best.pt")  # pretrained YOLOv8n model
file = cv2.imread("./infer/test2.jpg", cv2.IMREAD_UNCHANGED)
width = file.shape[1]
height = file.shape[0]
if width / 640 > height / 640:
    width, height = 640, int(height * 640 / width)
else:
    height, width = 640, int(width * 640 / height)
file = cv2.resize(file, (width, height), interpolation=cv2.INTER_AREA)
# Run batched inference on a list of images
results = model([file, file, file])  # return a list of Results objects
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)
