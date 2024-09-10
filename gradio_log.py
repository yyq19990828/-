import gradio as gr
import PIL.Image as Image
from ultralytics import ASSETS, YOLO
import os
import random

model = YOLO("/workspace/RDD_yolo/v8l_no_pre/0802_3classes_640/weights/best.pt")
pre_path = '/workspace/data/predict'
pre_path2 = '/workspace/data/new_combined_RDD_dataset/test/images'
def predict_image(img, conf_threshold, iou_threshold):
    """使用可调节的置信度和 IoU 阈值预测图像中的目标。"""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
        verbose=False
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

# Get the list of image files in the pre_path directory
image_files = [[os.path.join(pre_path, file), 0.15, 0.5] for file in os.listdir(pre_path) if file.endswith(".jpg")]

image_files2 = [[os.path.join(pre_path2, file), 0.25, 0.5] for file in random.sample(os.listdir(pre_path2), 10)]

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="上传图像"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="置信度阈值"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU 阈值"),
    ],
    outputs=gr.Image(type="pil", label="结果"),
    title="Ultralytics Gradio",
    # description="上传图像进行推理。默认使用 Ultralytics YOLOv8n 模型。",
    examples=image_files[2:]+image_files2,
)

if __name__ == "__main__":
    iface.launch(share=True)