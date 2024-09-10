from ultralytics import YOLO
from 飞行检测.model_args.args import parse_args

# 使用示例
args = parse_args()

#通用参数
args.model='/workspace/飞行检测/v8s_p2_new/0909_6cls_1024/weights/best.pt'
args.task = 'detect'
args.project = 'tyjt'
# args.name = 'finetune_0826_4cls_1024'#
args.exist_ok = True

#设备参数
args.device = [0]

#预测设置
args.source = r'/workspace/data/tyjt自采无人机/TYJT-CAMR-2024-09-02-16-50-25.mp4'
# args.source = r'/workspace/data/tyjt自采无人机/TYJT-CAMR-2024-08-30-17-16-03.mp4'
args.imgsz = 1024
args.conf = 0.1
args.iou = 0.2
# args.classes = [0, 1, 2, 3] # 按类别过滤结果，如classes=0, 或classes=[0,2,3]
args.visualize = False
args.save = False
args.show = True
args.save_frames = False
args.save_txt = False
args.save_conf = False
args.line_width = 0.1


model = YOLO(model=args.model, task=args.task)  # Load model

# Run inference on the source
model.predict(source=args.source, stream=False, show=args.show, save=args.save, imgsz=args.imgsz,
              project = args.project, exist_ok=args.exist_ok, name=args.name,
              conf=args.conf, max_det=100,iou=args.iou,#检测框相关
              vid_stride=1, #跳帧
              device=args.device,  # device=0为GPU，-1为CPU
            #   classes=[0],#类别相关
              # visualize=True, #特征可视化
              save_frames=args.save_frames, save_txt=args.save_txt #保存图片和txt
              )  # list of Results objects

#find / -type f -name "window.cpp"
# sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev
# sudo apt-get install libgtk2.0-dev
# sudo apt-get install pkg-config

# opencv-contrib-python   4.8.0.74
# opencv-fixer            0.2.5
# opencv-python           4.8.0.74
# opencv-python-headless  4.8.0.74

# >>> import cv2
# >>> cv2__version__
