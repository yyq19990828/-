from ultralytics import YOLO
from ultralytics.data import augment
from ultralytics.utils.loss import FocalLoss
import argparse

from model_args.args import parse_args
from custom_modules.custom_trainer import CustomTrainer

# 使用示例
args = parse_args()

#通用参数
args.model='/workspace/飞行检测/custom_model_config/yolov8s-ghost-onlyp2.yaml'
args.task = 'detect'
args.project = './results/fine_tune'
args.name = '0910_1cls_1024'
args.exist_ok = True
args.val = False

#设备参数
args.device = [0, 1]

#训练参数
args.data = '/workspace/飞行检测/data_config/finetune.yaml'
args.epochs = 1
args.batch = 16
args.imgsz = 1024
args.save = True
args.patience = 0
args.optimizer = 'Adam'
args.pretrained = True
args.label_smoothing = 0
args.freeze = 15
args.cache = True
args.fraction = 0.01

args.resume = False

args.box = 7.5
args.cls = 0.05
args.dfl = 1.5
args.lr0 = 0.001
args.lrf = 0.1
# args.warmup_epochs = 0
args.coslr = True

#数据增强参数
args.mosaic = 0.5
args.close_mosaic = 10
args.copy_paste = 0.3
args.mixup = 0.5
# args.hsv_h = 0.015
# args.hsv_s = 0.7
# args.hsv_v = 0.4
# args.degrees = 0.0
# args.translate = 0.1
# args.scale = 0.5
# args.shear = 0.0
# args.perspective = 0.0
# args.flipud = 0.0
# args.fliplr = 0.5
# args.bgr = 0.0

# YOLO
model = YOLO(model=args.model, task=args.task)  # Load model

# 加载backbone模型
backbone_ckpt = '/workspace/飞行检测/results/v8s_p2_new/0910_5cls_1024/weights/backbone_module_epoch0.pt'
model.load(weights=backbone_ckpt)

# 冻结backbone模型
model.train(trainer=CustomTrainer, data=args.data, pretrained=args.pretrained, epochs=args.epochs, batch=args.batch, device=args.device, 
            project=args.project, name=args.name, exist_ok=args.exist_ok, save=args.save, resume=args.resume,
            label_smoothing=args.label_smoothing, patience=args.patience, optimizer=args.optimizer, verbose=args.verbose, 
            imgsz=args.imgsz,mosaic=args.mosaic,close_mosaic=args.close_mosaic,copy_paste=args.copy_paste, mixup=args.mixup, # 增强超参数
            cos_lr=args.cos_lr, lr0=args.lr0, lrf=args.lrf, val=args.val,
            warmup_epochs=args.warmup_epochs,
            box=args.box, cls=args.cls, dfl=args.dfl, # Train model
            cache=args.cache, fraction=args.fraction, # Cache images for faster training
            )  

