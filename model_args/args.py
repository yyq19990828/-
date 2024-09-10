import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='UltraLytics cfg 参数')

    #创建参数组
    general_group = parser.add_argument_group('通用参数')
    train_group = parser.add_argument_group('训练参数')
    val_group = parser.add_argument_group('验证/测试参数')
    predict_group = parser.add_argument_group('预测设置')
    
    #通用参数
    general_group.add_argument('--model', type=str, default=None, help='指定用于训练的模型文件。接受 .pt 预训练模型或 .yaml 配置文件的路径。')
    general_group.add_argument('--task', type=str, default='detect', help='(str) YOLO任务类型，如detect、segment、classify、pose')
    general_group.add_argument('--project', type=str, help='(str, 可选) 项目名称') 
    general_group.add_argument('--name', type=str, help='(str, 可选) 实验名称，结果保存在project/name目录下')
    general_group.add_argument('--exist_ok', type=bool, default=False, help='(bool) 是否覆盖已有的实验结果')
    
    #训练参数
    train_group.add_argument('--data', type=str, help='(str, 可选) 数据文件路径，如coco8.yaml')
    train_group.add_argument('--epochs', type=int, default=100, help='(int) 训练的epoch数量')
    train_group.add_argument('--time', type=float, help='(float, 可选) 训练的小时数，如果提供则会覆盖epochs参数')
    train_group.add_argument('--patience', type=int, default=100, help='(int) early stopping等待的epoch数量，如果没有观察到改进则停止训练')
    train_group.add_argument('--batch', type=int, default=16, help='(int) 每个batch的图像数量（-1表示自动确定batch size）')
    train_group.add_argument('--imgsz', type=int, default=640, help='(int | list) 训练和验证模式下的输入图像尺寸，预测和导出模式下可以是[h,w]形式的列表')
    train_group.add_argument('--save', type=bool, default=True, help='(bool) 是否保存训练的checkpoint和预测结果')
    train_group.add_argument('--save_period', type=int, default=-1, help='(int) 每隔多少个epoch保存一次checkpoint（如果小于1则禁用(*)')
    train_group.add_argument('--cache', type=bool, default=False, help='(bool) 是否使用缓存加载数据，可选值为True/ram、disk或False')
    train_group.add_argument('--device', help='(int | str | list, 可选) 运行设备，如device=0或device=0,1,2,3或device=cpu')
    train_group.add_argument('--workers', type=int, default=8, help='(int) 数据加载的worker线程数（如果使用DDP，则是每个RANK的线程数）')
    train_group.add_argument('--pretrained', type=str, default=True, help='(bool | str) 是否使用预训练模型（bool值）或加载权重的模型路径（str值）')
    train_group.add_argument('--optimizer', type=str, default='auto', choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'], help='(str) 使用的优化器，可选值为SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto')
    train_group.add_argument('--verbose', type=bool, default=True, help='(bool) 是否打印详细输出')
    train_group.add_argument('--seed', type=int, default=0, help='(int) 随机数种子，用于结果可复现') 
    train_group.add_argument('--deterministic', type=bool, default=True, help='(bool) 是否启用确定性模式')
    train_group.add_argument('--single_cls', type=bool, default=False, help='(bool) 是否将多类别数据作为单类别训练')
    train_group.add_argument('--rect', type=bool, default=False, help='(bool) 是否使用矩形训练，在mode为train时生效。在mode为val时控制是否使用矩形验证')
    train_group.add_argument('--cos_lr', type=bool, default=False, help='(bool) 是否使用cosine学习率调度器')  
    train_group.add_argument('--close_mosaic', type=int, default=10, help='(int) 在最后几个epoch禁用mosaic数据增强（0表示不禁用）')
    train_group.add_argument('--resume', type=bool, default=False, help='(bool) 是否从最后一个checkpoint恢复训练') 
    train_group.add_argument('--amp', type=bool, default=True, help='(bool) 是否使用自动混合精度(AMP)训练，可选值为True或False。设为True会运行AMP检查')
    train_group.add_argument('--fraction', type=float, default=1.0, help='(float) 用于训练的数据集比例（默认为1.0，即使用所有训练图像）')
    train_group.add_argument('--profile', type=bool, default=False, help='(bool) 在训练期间对ONNX和TensorRT速度进行性能分析以生成日志')
    train_group.add_argument('--freeze', help='(int | list, 可选) 在训练期间冻结前n层或由索引列表指定的层')


    # 验证/测试设置  
    val_group.add_argument('--val', type=bool, default=True, help='(bool) 是否在训练期间进行验证/测试')
    val_group.add_argument('--split', type=str, default='val', help='(str) 用于验证的数据集split，如val、test或train') 
    val_group.add_argument('--save_json', type=bool, default=False, help='(bool) 是否将结果保存为JSON文件')
    val_group.add_argument('--save_hybrid', type=bool, default=False, help='(bool) 是否保存标签的混合版本（标签+额外预测）')
    val_group.add_argument('--conf', type=float, help='(float, 可选) 检测的置信度阈值（默认为预测时0.25，验证时0.001）')
    val_group.add_argument('--iou', type=float, default=0.7, help='(float) 非极大值抑制(NMS)的IoU阈值') 
    val_group.add_argument('--max_det', type=int, default=300, help='(int) 每张图像的最大检测数量')
    val_group.add_argument('--half', type=bool, default=False, help='(bool) 是否使用半精度(FP16)')
    val_group.add_argument('--dnn', type=bool, default=False, help='(bool) 是否使用OpenCV DNN进行ONNX推理')
    val_group.add_argument('--plots', type=bool, default=True, help='(bool) 在训练/验证期间保存图像图表')

    # 预测设置
    predict_group = parser.add_argument_group('预测设置')
    predict_group.add_argument('--source', help='(str, 可选) 图像或视频的源目录') 
    predict_group.add_argument('--vid_stride', type=int, default=1, help='(int) 视频帧率stride')
    predict_group.add_argument('--stream_buffer', type=bool, default=False, help='(bool) 缓冲所有流帧(True)或仅返回最新帧(False)')
    predict_group.add_argument('--visualize', type=bool, default=False, help='(bool) 可视化模型特征')
    predict_group.add_argument('--augment', type=bool, default=False, help='(bool) 对预测源应用图像增强')  
    predict_group.add_argument('--agnostic_nms', type=bool, default=False, help='(bool) 类别不可知的NMS') 
    predict_group.add_argument('--classes', help='(int | list[int], 可选) 按类别过滤结果，如classes=0, 或classes=[0,2,3]')
    predict_group.add_argument('--retina_masks', type=bool, default=False, help='(bool) 使用高分辨率分割掩码') 
    predict_group.add_argument('--embed', help='(list[int], 可选) 从给定层返回特征向量/嵌入')

    # 可视化设置
    vis_group = parser.add_argument_group('可视化设置(预测)')
    vis_group.add_argument('--show', type=bool, default=False, help='(bool) 如果环境允许，显示预测的图像和视频')
    vis_group.add_argument('--save_frames', type=bool, default=False, help='(bool) 保存预测的单独视频帧')
    vis_group.add_argument('--save_txt', type=bool, default=False, help='(bool) 将结果保存为.txt文件') 
    vis_group.add_argument('--save_conf', type=bool, default=False, help='(bool) 将结果和置信度分数一起保存')
    vis_group.add_argument('--save_crop', type=bool, default=False, help='(bool) 将裁剪后的图像与结果一起保存') 
    vis_group.add_argument('--show_labels', type=bool, default=True, help='(bool) 显示预测标签，如person') 
    vis_group.add_argument('--show_conf', type=bool, default=True, help='(bool) 显示预测置信度，如0.99')
    vis_group.add_argument('--show_boxes', type=bool, default=True, help='(bool) 显示预测框')
    vis_group.add_argument('--line_width', help='(int, 可选) 检测边框的线宽。如果为None，将按图像大小缩放')

    # 导出设置 
    export_group = parser.add_argument_group('导出设置')
    export_group.add_argument('--format', type=str, default='torchscript', help='(str) 导出格式，可选值参见 https://docs.ultralytics.com/modes/export/#export-formats')
    export_group.add_argument('--keras', type=bool, default=False, help='(bool) 是否使用Keras导出')  
    export_group.add_argument('--optimize', type=bool, default=False, help='(bool) TorchScript：是否为移动设备进行优化')
    export_group.add_argument('--int8', type=bool, default=False, help='(bool) CoreML/TF INT8 量化') 
    export_group.add_argument('--dynamic', type=bool, default=False, help='(bool) ONNX/TF/TensorRT: 是否使用动态轴') 
    export_group.add_argument('--simplify', type=bool, default=False, help='(bool) ONNX: 使用onnxslim简化模型')
    export_group.add_argument('--opset', type=int, help='(int, 可选) ONNX: 指定opset版本') 
    export_group.add_argument('--workspace', type=int, default=4, help='(int) TensorRT: workspace大小 (GB)') 
    export_group.add_argument('--nms', type=bool, default=False, help='(bool) CoreML: 添加NMS') 

    # 超参数设置
    hyp_group = parser.add_argument_group('超参数设置') 
    hyp_group.add_argument('--lr0', type=float, default=0.01, help='(float) 初始学习率 (SGD=1E-2, Adam=1E-3)')
    hyp_group.add_argument('--lrf', type=float, default=0.01, help='(float) 最终OneCycleLR学习率 (lr0 * lrf)') 
    hyp_group.add_argument('--momentum', type=float, default=0.937, help='(float) SGD momentum/Adam beta1')
    hyp_group.add_argument('--weight_decay', type=float, default=0.0005, help='(float) optimizer weight decay 5e-4')
    hyp_group.add_argument('--warmup_epochs', type=float, default=3.0, help='(float) warmup epochs (fractions ok)') 
    hyp_group.add_argument('--warmup_momentum', type=float, default=0.8, help='(float) warmup initial momentum') 
    hyp_group.add_argument('--warmup_bias_lr', type=float, default=0.1, help='(float) warmup initial bias lr')
    hyp_group.add_argument('--box', type=float, default=7.5, help='(float) box loss gain') 
    hyp_group.add_argument('--cls', type=float, default=0.5, help='(float) cls loss gain (scale with pixels)')
    hyp_group.add_argument('--dfl', type=float, default=1.5, help='(float) dfl loss gain') 
    hyp_group.add_argument('--pose', type=float, default=12.0, help='(float) pose loss gain')
    hyp_group.add_argument('--kobj', type=float, default=1.0, help='(float) keypoint obj loss gain') 
    hyp_group.add_argument('--label_smoothing', type=float, default=0.0, help='(float) label smoothing (fraction)')
    hyp_group.add_argument('--nbs', type=int, default=64, help='(int) nominal batch size') 
    hyp_group.add_argument('--hsv_h', type=float, default=0.015, help='图像HSV-色调增强(分数)')  
    hyp_group.add_argument('--hsv_s', type=float, default=0.7, help='图像HSV-饱和度增强(分数)')
    hyp_group.add_argument('--hsv_v', type=float, default=0.4, help='图像HSV-明度增强(分数)')
    hyp_group.add_argument('--degrees', type=float, default=0.0, help='图像旋转(正负度数)')
    hyp_group.add_argument('--translate', type=float, default=0.1, help='图像平移(正负分数)')
    hyp_group.add_argument('--scale', type=float, default=0.5, help='图像缩放(正负增益)')
    hyp_group.add_argument('--shear', type=float, default=0.0, help='图像剪切(正负度数)')  
    hyp_group.add_argument('--perspective', type=float, default=0.0, help='图像透视(正负分数),范围0-0.001')
    hyp_group.add_argument('--flipud', type=float, default=0.0, help='图像上下翻转(概率)')
    hyp_group.add_argument('--fliplr', type=float, default=0.5, help='图像左右翻转(概率)')
    hyp_group.add_argument('--bgr', type=float, default=0.0, help='图像通道BGR(概率)')
    hyp_group.add_argument('--mosaic', type=float, default=1.0, help='图像马赛克增强(概率)') 
    hyp_group.add_argument('--mixup', type=float, default=0.0, help='图像mixup增强(概率)')
    hyp_group.add_argument('--copy_paste', type=float, default=0.0, help='分割区域copy-paste增强(概率)')

    return parser.parse_args()

# 使用示例
args = parse_args()

#通用参数
args.model='/workspace/...'
args.task = 'detect'
args.project = 'v8s_pre'
args.name = '0821_4classes_512v1'
args.exist_ok = True

args.device = [0, 1]

#训练参数
args.data = '/workspace/...'
args.epochs = 200
args.batch = 64
args.imgsz = 640
args.save = True
args.patience = 20
args.optimizer = 'AdamW'
args.pretrained = True
args.label_smoothing = 0.1

args.resume = False

args.box = 7.5
args.cls = 1.5
args.dfl = 1.5
args.lr0 = 0.001
args.lrf = 0.1
args.warmup_epochs = 3
args.coslr = True

args.mosaic = 0.5
args.close_mosaic = 20
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

#预测设置
args.source = '/workspace/...'
# args.classes = [0, 1, 2, 3] # 按类别过滤结果，如classes=0, 或classes=[0,2,3]
args.visualize = False
args.save_frames = True
args.save_txt = False
args.save_conf = False

#导出设置
args.format = 'onnx'
args.opset = None

# print(args)  # 访问参数



