#!/bin/bash

# 设置要运行的Python文件路径
py_file="/workspace/飞行检测/train.py"
model="/workspace/飞行检测/models/Pretrained_Drone/yolov8s.pt"
model_new="yolov8s.yaml"
project1="v8s_pre"
project2="v8s_new"


# 设置要运行的次数
n=2

# 定义一个数组,存储每次运行时的命令行参数
# 比较图片的降采样大小和是否预训练
# model
# project
# name
# epochs
# pretrained
# lr0
# lrf

args=(
  # "--model $model --project ${project1} --name 0823_4cls_512 --epochs 50 --lr0 0.0001 --lrf 0.1 --warmup_epochs 0 --imgsz 512"
  # "--model ${model} --project ${project1} --name 0823_4cls_1024 --epochs 50 --lr0 0.0001 --lrf 0.1 --warmup_epochs 0 --imgsz 1024"
  "--model ${model_new} --project ${project2} --name 0826_4cls_512 --epochs 60 --lr0 0.001 --lrf 0.1 --warmup_epochs 3 --imgsz 512"
  "--model ${model_new} --project ${project2} --name 0826_4cls_1024 --epochs 60 --lr0 0.001 --lrf 0.1 --warmup_epochs 3 --imgsz 1024"
#   "--model"
)

# 循环运行Python文件n次
for ((i=0; i<n; i++)); do
  # 获取当前次的命令行参数
  current_args=${args[$i]}
  
  # 运行Python文件,传递当前次的命令行参数
  python "$py_file" $current_args
  
  echo "Finished run $((i+1)) with args: $current_args"
done

echo "All runs completed."
