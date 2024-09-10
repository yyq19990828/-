from ultralytics.nn.modules import Detect
from ultralytics import YOLO
from ultralytics.nn.modules import head

# Load a model
# model = YOLO("/workspace/RDD_yolo/models/yolov8n.pt")  # load an official model
model = YOLO("/workspace/飞行检测/fine_tune/0830_6cls_1024/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx", opset=17)
# model.export(format="engine")