# %%
img_path = '/media/tyjt/rog/train/images'
label_path = '/media/tyjt/rog/train/labels'

# %% [markdown]
# 给定txt标注文件目录，遍历当前目录下的txt文件查看是否为空，若为空则删除，并打印文件名字

# %%
import os

def remove_empty_txt_files(current_directory):
    # current_directory = os.getcwd()
    for file in os.listdir(current_directory):
        if file.endswith(".txt"):
            file_path = os.path.join(current_directory, file)
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                print(f"Removed empty file: {file}")

# 调用函数删除空的txt文件
path = r'/media/tyjt/rog/train/labels'
remove_empty_txt_files(path)

# %% [markdown]
# 给定图像目录和txt目录，遍历图像文件，如果前缀名没有出现在txt目录中，则将该图片文件删除

# %%
import os

def remove_images_without_txt(image_directory, txt_directory):
    txt_files = [file.split(".")[0] for file in os.listdir(txt_directory) if file.endswith(".txt")]

    for file in os.listdir(image_directory):
        if file.endswith((".jpg", ".jpeg", ".png")):
            image_prefix = file.split(".")[0]
            if image_prefix not in txt_files:
                image_path = os.path.join(image_directory, file)
                os.remove(image_path)
                print(f"Removed image: {file}")

# 指定图像目录和txt目录
image_directory = "/workspace/data/飞行物体(0.01%)/dots/images"
txt_directory = "/workspace/data/飞行物体(0.01%)/dots/labels"

# 调用函数删除没有对应txt文件的图像
remove_images_without_txt(image_directory, txt_directory)

# %%
len(os.listdir(img_path))

# %% [markdown]
# 给定图像目录和yolo格式的txt目录，遍历txt目录下的txt文件，如果txt文件中存在标签索引为1的信息，则将该txt文件移动到新的指定的label文件夹，同时将关联的图片文件移动到指定的image文件夹，这些文件夹可以允许新建，当txt文件中有多个目标时，打印该文件存在多个目标的信息。
# 
# 以下是一个Python脚本，根据给定的图像目录和YOLO格式的txt目录，遍历txt文件，如果文件中存在标签索引为1的信息，则将该txt文件和关联的图片文件移动到指定的文件夹，并处理多个目标的情况：
# 
# 脚本说明：
# 
# 1. 导入 `os` 和 `shutil` 模块，用于处理文件和目录操作。
# 
# 2. 定义 `move_files_with_label_1` 函数，接受四个参数：`image_directory` 表示图像目录的路径，`txt_directory` 表示txt目录的路径，`label_directory` 表示新的标签文件夹的路径，`new_image_directory` 表示新的图像文件夹的路径。
# 
# 3. 使用 `os.makedirs` 函数创建标签文件夹和图像文件夹（如果不存在）。
# 
# 4. 遍历txt目录中的所有文件：
#    - 如果文件以 ".txt" 结尾，则认为它是一个txt文件。
#    - 构造txt文件的完整路径 `txt_path` 和对应图像文件的完整路径 `image_path`。
#    - 检查图像文件是否存在。
# 
# 5. 如果图像文件存在：
#    - 读取txt文件的内容，并逐行检查是否存在标签索引为1的信息。
#    - 如果找到标签索引为1的信息，则将 `label_found` 标志设置为 `True`。
#    - 如果txt文件中包含多个目标，则将 `multiple_objects` 标志设置为 `True`。
# 
# 6. 如果找到标签索引为1的信息：
#    - 构造新的txt文件路径 `new_txt_path` 和新的图像文件路径 `new_image_path`。
#    - 使用 `shutil.move` 函数将txt文件和图像文件移动到新的文件夹中。
#    - 打印移动文件的信息。
# 
# 7. 如果存在多个目标，则打印相关信息。
# 
# 8. 如果图像文件不存在，则打印文件不存在的信息。
# 
# 9. 指定要遍历的图像目录、txt目录、新的标签文件夹和新的图像文件夹的路径。
# 
# 10. 调用 `move_files_with_label_1` 函数，传入相应的路径，开始执行文件移动操作。
# 
# 请确保将图像目录、txt目录、新的标签文件夹和新的图像文件夹的路径替换为实际的路径。运行脚本后，它将遍历txt目录中的所有txt文件，如果文件中存在标签索引为1的信息，则将该txt文件和关联的图像文件移动到指定的文件夹中。如果txt文件中包含多个目标，则会打印相关信息。

# %%
import os
import shutil

def move_files_with_label_1(image_directory, txt_directory, label_directory, new_image_directory):
    # 创建标签文件夹和图像文件夹（如果不存在）
    os.makedirs(label_directory, exist_ok=True)
    os.makedirs(new_image_directory, exist_ok=True)

    for file in os.listdir(txt_directory):
        if file.endswith(".txt"):
            txt_path = os.path.join(txt_directory, file)
            image_prefix = file.split(".")[0]
            image_path = os.path.join(image_directory, image_prefix + ".jpg")

            # 检查图像文件是否存在
            if os.path.exists(image_path):
                label_found = False
                multiple_objects = False

                # 读取txt文件内容
                with open(txt_path, "r") as f:
                    lines = f.readlines()

                # 检查是否存在标签索引为1的信息
                for line in lines:
                    label_index = line.split()[0]
                    if label_index == "1":
                        label_found = True
                    if len(lines) > 1:
                        multiple_objects = True

                # 如果找到标签索引为1的信息，则移动文件
                if label_found:
                    new_txt_path = os.path.join(label_directory, file)
                    new_image_path = os.path.join(new_image_directory, image_prefix + ".jpg")
                    shutil.move(txt_path, new_txt_path)
                    shutil.move(image_path, new_image_path)
                    print(f"Moved {file} and {image_prefix}.jpg to new directories.")

                    # 如果存在多个目标，则打印信息
                    if multiple_objects:
                        print(f"{file} contains multiple objects.")
            else:
                print(f"Image file {image_prefix}.jpg not found.")

# 指定图像目录、txt目录、标签文件夹和新图像文件夹
# image_directory = "/path/to/image/directory"
# txt_directory = "/path/to/txt/directory"
label_directory = "Drone_only/labels"
new_image_directory = "Drone_only/images"

# 调用函数移动文件
move_files_with_label_1(img_path, label_path, label_directory, new_image_directory)

# %% [markdown]
# 给定图像目录和yolo格式的txt目录，遍历txt目录下的txt文件，如果txt文件的前缀名以V_DRONE开头，则将该txt文件移动到新的指定的label文件夹，同时将关联的图片文件移动到指定的image文件夹，这些文件夹可以允许新建。

# %%
import os
import shutil

def move_files(image_dir, txt_dir, new_label_dir, new_image_dir):
    # 创建新的label和image文件夹（如果不存在）
    os.makedirs(new_label_dir, exist_ok=True)
    os.makedirs(new_image_dir, exist_ok=True)

    # 遍历txt目录下的所有txt文件
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            # 获取txt文件的前缀名
            prefix = txt_file.split('.')[0]
            
            # 检查前缀名是否以V_DRONE开头
            if prefix.startswith('V_HELICOPTER'):
                # 构建关联的图片文件名
                image_file = prefix + '.jpg'  # 假设图片文件的扩展名为.jpg
                
                # 移动txt文件到新的label文件夹
                shutil.move(os.path.join(txt_dir, txt_file), os.path.join(new_label_dir, txt_file))
                
                # 移动关联的图片文件到新的image文件夹
                if os.path.exists(os.path.join(image_dir, image_file)):
                    shutil.move(os.path.join(image_dir, image_file), os.path.join(new_image_dir, image_file))
                else:
                    print(f"警告: 找不到对应的图片文件 {image_file}")

# 示例用法
image_directory = '/media/tyjt/rog/train/images'
txt_directory = '/media/tyjt/rog/train/labels_old'
new_label_directory = '/media/tyjt/rog/train/V_HELICOPTER/labels'
new_image_directory = '/media/tyjt/rog/train/V_HELICOPTER/images'

move_files(image_directory, txt_directory, new_label_directory, new_image_directory)

# %%
import os
len(os.listdir('/media/tyjt/rog/train/images')), len(os.listdir('/media/tyjt/rog/train/labels_old'))

# %% [markdown]
# 给定图像目录和yolo格式的txt目录，遍历txt目录下的txt文件，如果txt文件的前缀名以V_HELICOPTER_057开头，则将该txt文件删除，同时将关联的图片文件删除除。

# %%
import os

def delete_files(image_dir, txt_dir):
    # 遍历txt目录下的所有txt文件
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            # 获取txt文件的前缀名
            prefix = txt_file.split('.')[0]
            
            # 检查前缀名是否以V_HELICOPTER_057开头
            if prefix.startswith('pic'):
                # 构建关联的图片文件名
                image_file = prefix + '.jpg'  # 假设图片文件的扩展名为.jpg
                
                # 删除txt文件
                os.remove(os.path.join(txt_dir, txt_file))
                
                # 删除关联的图片文件（如果存在）
                image_path = os.path.join(image_dir, image_file)
                if os.path.exists(image_path):
                    os.remove(image_path)
                else:
                    print(f"警告: 找不到对应的图片文件 {image_file}")

# 示例用法
image_directory = '/media/tyjt/rog/train/Drone(small)/images'
txt_directory = '/media/tyjt/rog/train/Drone(small)/labels'

delete_files(image_directory, txt_directory)

# %% [markdown]
# 给定图像目录和yolo格式的txt目录，遍历txt目录下的txt文件，统计标注框小于原图3%的目标的比例

# %%
import os
from PIL import Image

# def calculate_small_object_ratio(image_dir, txt_dir):
#     total_objects = 0
#     small_objects = 0

#     # 遍历txt目录下的所有txt文件
#     for txt_file in os.listdir(txt_dir):
#         if txt_file.endswith('.txt'):
#             # 获取txt文件的前缀名
#             prefix = txt_file.split('.')[0]
            
#             # 构建关联的图片文件名
#             image_file = prefix + '.jpg'  # 假设图片文件的扩展名为.jpg
#             image_path = os.path.join(image_dir, image_file)
            
#             # 检查对应的图片文件是否存在
#             if os.path.exists(image_path):
#                 # 读取图片尺寸
#                 with Image.open(image_path) as img:
#                     width, height = img.size
                
#                 # 读取txt文件中的标注信息
#                 with open(os.path.join(txt_dir, txt_file), 'r') as f:
#                     lines = f.readlines()
                
#                 # 统计标注框小于原图3%的目标数量
#                 for line in lines:
#                     total_objects += 1
#                     data = line.strip().split()
#                     bbox_width = float(data[3]) * width
#                     bbox_height = float(data[4]) * height
#                     bbox_area = bbox_width * bbox_height
#                     image_area = width * height
                    
#                     if bbox_area / image_area < 0.03:
#                         small_objects += 1
#             else:
#                 print(f"警告: 找不到对应的图片文件 {image_file}")
    
#     # 计算标注框小于原图3%的目标的比例
#     if total_objects > 0:
#         small_object_ratio = small_objects / total_objects
#         print(f"标注框小于原图3%的目标的比例: {small_object_ratio:.2%}")
#     else:
#         print("没有找到有效的标注信息")

def calculate_small_object_ratio(txt_dir):
    total_objects = 0
    small_objects = 0

    # 遍历txt目录下的所有txt文件
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            # 读取txt文件中的标注信息
            with open(os.path.join(txt_dir, txt_file), 'r') as f:
                lines = f.readlines()
            
            # 统计标注框小于原图3%的目标数量
            for line in lines:
                total_objects += 1
                data = line.strip().split()
                bbox_width = float(data[3])
                bbox_height = float(data[4])
                bbox_area = bbox_width * bbox_height
                
                if bbox_area < 0.0001:
                    small_objects += 1
    
    # 计算标注框小于原图3%的目标的比例
    if total_objects > 0:
        small_object_ratio = small_objects / total_objects
        print(f"标注框小于原图3%的目标的比例: {small_object_ratio:.2%}")
    else:
        print("没有找到有效的标注信息")

# 示例用法
image_directory = '/workspace/data/evtol/images'
txt_directory = '/workspace/data/evtol/labels'

calculate_small_object_ratio(txt_directory)

# %% [markdown]
# 给定图像目录和yolo格式的txt目录，遍历txt目录下的txt文件，统计标注框小于原图3%的目标的比例，如果一张图上所有的目标都大于3%，把txt文件和相关联的图片移动到新的image和label文件夹

# %%
import os
import shutil

def process_files(image_dir, txt_dir, new_image_dir, new_label_dir):
    total_objects = 0
    small_objects = 0

    # 创建新的image和label文件夹（如果不存在）
    os.makedirs(new_image_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)

    # 遍历txt目录下的所有txt文件
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            # 获取txt文件的前缀名
            prefix = '.'.join(txt_file.split('.')[0:-1])

            # 构建关联的图片文件名
            image_file = prefix + '.jpg'  # 假设图片文件的扩展名为.jpg
            image_path = os.path.join(image_dir, image_file)
            
            # 检查对应的图片文件是否存在
            if os.path.exists(image_path):
                # 读取txt文件中的标注信息
                with open(os.path.join(txt_dir, txt_file), 'r') as f:
                    lines = f.readlines()
                
                # 统计标注框小于原图3%的目标数量
                small_object_count = 0
                for line in lines:
                    total_objects += 1
                    data = line.strip().split()
                    bbox_width = float(data[3])
                    bbox_height = float(data[4])
                    bbox_area = bbox_width * bbox_height
                    
                    if bbox_area < 0.005:
                        small_objects += 1
                        small_object_count += 1
                
                # 如果所有目标都大于3%，则移动图片和txt文件到新的文件夹
                if small_object_count == 0:
                    shutil.move(image_path, os.path.join(new_image_dir, image_file))
                    shutil.move(os.path.join(txt_dir, txt_file), os.path.join(new_label_dir, txt_file))
            else:
                print(f"警告: 找不到对应的图片文件 {image_file}")
    
    # 计算标注框小于原图3%的目标的比例
    if total_objects > 0:
        small_object_ratio = small_objects / total_objects
        print(f"标注框小于原图0.5%的目标的比例: {small_object_ratio:.2%}")
    else:
        print("没有找到有效的标注信息")

# 示例用法
image_directory = '/workspace/data/飞行物体汇总/V_PLANE(small)/images'
txt_directory = '/workspace/data/飞行物体汇总/V_PLANE(small)/labels'
new_image_directory = '/workspace/data/飞行物体汇总/V_PLANE(middle)/images'
new_label_directory = '/workspace/data/飞行物体汇总/V_PLANE(middle)/labels'

process_files(image_directory, txt_directory, new_image_directory, new_label_directory)

# %% [markdown]
# 

# %%
import os
import shutil

def process_files(dataset_dir, output_dir):
    # 创建输出文件夹
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')

    total_targets = 0
    small_targets = 0

    for txt_file in os.listdir(label_dir):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(label_dir, txt_file)
            image_path = os.path.join(image_dir, txt_file[:-4] + '.jpg')

            if not os.path.exists(image_path):
                continue

            # 读取图像尺寸
            with open(image_path, 'rb') as f:
                f.seek(0, 2)  # 移动到文件末尾
                image_size = f.tell()

            # 读取txt文件中的标注信息
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            targets = []
            for line in lines:
                _, _, _, w, h = map(float, line.strip().split())
                target_size = w * h
                targets.append(target_size)
                total_targets += 1
                if target_size < 0.0001:
                    small_targets += 1

            # 判断是否所有目标都大于3%
            if all(size < 0.0001 for size in targets):
                # 移动图片和txt文件到新文件夹
                shutil.move(image_path, os.path.join(output_image_dir, txt_file[:-4] + '.jpg'))
                shutil.move(txt_path, os.path.join(output_label_dir, txt_file))

    # 计算小目标的比例
    small_target_ratio = small_targets / total_targets if total_targets > 0 else 0

    print(f"Small target ratio: {small_target_ratio:.2%}")
    print("Files moved successfully.")

# 指定数据集目录和输出目录
dataset_directory = '/workspace/data/unkown(drone)'
output_directory = '/workspace/data/飞行物体(0.01%)/Drone(tiny)'

# 处理文件
process_files(dataset_directory, output_directory)

# %% [markdown]
# 给定一个带yolo标注的文件目录，遍历所有的txt文件，标签索引改为指定的值

# %%
import os

def change_label_index(txt_dir, new_label_index):
    # 遍历txt目录及其子目录下的所有txt文件
    for root, dirs, files in os.walk(txt_dir):
        for txt_file in files:
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(root, txt_file)
                
                # 读取txt文件中的标注信息
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                
                # 更新标签索引
                updated_lines = []
                for line in lines:
                    data = line.strip().split()
                    data[0] = str(new_label_index)
                    updated_line = ' '.join(data) + '\n'
                    updated_lines.append(updated_line)
                
                # 将更新后的标注信息写回txt文件
                with open(txt_path, 'w') as f:
                    f.writelines(updated_lines)
    
    print("标签索引已更新完成")

# 示例用法
txt_directory = '/workspace/自建数据集/data(0.01%)'
new_label_index = 0  # 指定新的标签索引值

change_label_index(txt_directory, new_label_index)

# %% [markdown]
# 给定一个目录，查看该目录下是否存在labels的文件夹，若存在，根据yolo格式的txt文件(已经归一化)，统计目标大小占比的分布，画图表示，假设目标没有大于5%，在0到5%中统计，以0.1%为标准

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_yolo_labels(directory):
    labels_dir = os.path.join(directory, 'labels')
    
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found in {directory}")
        return None

    size_ratios = []

    for file in os.listdir(labels_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(labels_dir, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    _, _, _, width, height = map(float, line.strip().split())
                    size_ratio = width * height * 100
                    size_ratios.append(size_ratio)

    if len(size_ratios) == 0:
        print("No valid YOLO labels found.")
        return None

    bins = np.arange(0, 5.1, 0.1)
    hist, _ = np.histogram(size_ratios, bins=bins)

    return hist, bins

def plot_distribution(hist, bins):
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], hist, width=0.1, align='edge')
    plt.xlabel('Object Size Ratio (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Object Size Ratios')
    plt.xticks(np.arange(0, 5.1, 0.5))
    plt.grid(True)
    plt.show()

# 指定目录路径
directory = '/workspace/data/evtol'

# 分析YOLO标注文件
result = analyze_yolo_labels(directory)

if result is not None:
    hist, bins = result
    plot_distribution(hist, bins)

# %% [markdown]
# 将一个指定目录下的labels文件夹和images文件夹内的文件移动到另一个指定文件夹的同门文件夹下，如果有名字重复则替换

# %%
import os
import shutil

def move_files(src_dir, dst_dir):
    # 移动labels文件夹内的文件
    src_labels_dir = os.path.join(src_dir, 'labels')
    dst_labels_dir = os.path.join(dst_dir, 'labels')
    
    if not os.path.exists(dst_labels_dir):
        os.makedirs(dst_labels_dir)
    
    for file_name in os.listdir(src_labels_dir):
        src_file = os.path.join(src_labels_dir, file_name)
        dst_file = os.path.join(dst_labels_dir, file_name)
        shutil.move(src_file, dst_file)
    
    # 移动images文件夹内的文件
    src_images_dir = os.path.join(src_dir, 'images')
    dst_images_dir = os.path.join(dst_dir, 'images')
    
    if not os.path.exists(dst_images_dir):
        os.makedirs(dst_images_dir)
    
    for file_name in os.listdir(src_images_dir):
        src_file = os.path.join(src_images_dir, file_name)
        dst_file = os.path.join(dst_images_dir, file_name)
        shutil.move(src_file, dst_file)
    
    print("Files moved successfully.")

# 指定源目录和目标目录
source_directory = ''#'/workspace/data/飞行物体(>0.5%)/V_DRONE(middle)'
destination_directory = ''#'/workspace/data/飞行物体(>0.5%)/Drone(big)'

# 移动文件
move_files(source_directory, destination_directory)

# %% [markdown]
# 把目录下的png文件无损转为jpg文件

# %%
import os
from PIL import Image

def convert_png_to_jpg(directory):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # 构建PNG文件的完整路径
            png_path = os.path.join(directory, filename)
            
            # 构建JPG文件的完整路径
            jpg_path = os.path.join(directory, filename[:-4] + '.jpg')
            
            # 打开PNG图像
            with Image.open(png_path) as img:
                # 将PNG图像转换为RGB模式
                rgb_img = img.convert('RGB')
                
                # 以最高质量保存为JPG格式
                rgb_img.save(jpg_path, 'JPEG', quality=100)
            
            print(f"Converted {filename} to JPG.")

# 指定要转换PNG文件的目录
directory_path = '/workspace/data/飞行物体(0.01%)/dots/images'

# 转换PNG文件为JPG文件
convert_png_to_jpg(directory_path)


