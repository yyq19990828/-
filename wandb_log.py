import wandb

wandb.login()
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO

# 初始化一个 Weights & Biases 运行
wandb.init(project="ultralytics", job_type="training")

# 加载一个 YOLO 模型
model = YOLO(model='yolov8l.yaml', task='detect')


# 为 Ultralytics 添加 W&B 回调
add_wandb_callback(model, enable_model_checkpointing=True)

# # 训练和微调模型
# model.train(project="ultralytics", data="coco8.yaml", epochs=5, imgsz=640)
model.train(data='/workspace/data/new_combined_RDD_dataset/data.yaml', pretrained=False, epochs=200, batch=32, device=[0, 1], 
            project='v8l_no_pre', name='0802_3classes_640', 
            label_smoothing=0, patience=20, exist_ok=False, optimizer='AdamW', verbose=False, copy_paste=.3,
            imgsz=640,mosaic=0.5,close_mosaic=50,
            cos_lr=True, lr0=0.001, lrf=0.1,
            warmup_epochs=3,
            box=7.5, cls=1.5, dfl=1.5)  # Load model

# # 验证模型
# model.val()

# 执行推理并记录结果
model.predict(source='/workspace/data/new_combined_RDD_dataset/test/images', batch=32, save=True, conf=0.2, device=[1], #, project='./v8m', name='0725_pre',
             iou = 0.3)

# 完成 W&B 运行
wandb.finish()