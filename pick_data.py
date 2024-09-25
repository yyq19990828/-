import os
import random
import shutil

def collect_random_samples(root_dir, output_dir, num_samples):
    # 删除已存在的输出文件夹（如果存在）
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # 创建输出文件夹
    os.makedirs(os.path.join(output_dir, 'train', 'images'))
    os.makedirs(os.path.join(output_dir, 'train', 'labels'))
    os.makedirs(os.path.join(output_dir, 'val', 'images'))
    os.makedirs(os.path.join(output_dir, 'val', 'labels'))
    
    # 遍历根目录下的所有子目录
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        
        # 检查子目录是否包含labels和images文件夹
        labels_dir = os.path.join(subdir_path, 'labels')
        images_dir = os.path.join(subdir_path, 'images')
        
        if os.path.isdir(labels_dir) and os.path.isdir(images_dir):
            # 获取labels文件夹中的所有txt文件
            txt_files = [file for file in os.listdir(labels_dir) if file.endswith('.txt')]
            
            # 随机挑选指定张数的图片
            if len(txt_files) > num_samples:
                selected_files = random.sample(txt_files, num_samples)
            else:
                selected_files = txt_files
            train_files = selected_files[:int(num_samples * 0.9)]
            val_files = selected_files[int(num_samples * 0.9):]
            
            print(f"Found {len(txt_files)} txt files in {subdir}: picking {len(selected_files)} files")
            # 复制选中的图片和对应的txt文件到输出文件夹
            for file in train_files:
                image_file = file.replace('.txt', '.jpg')  # 假设图片文件的扩展名为.jpg
                
                src_label_path = os.path.join(labels_dir, file)
                dst_label_path = os.path.join(output_dir, 'train', 'labels', file)
                shutil.copy(src_label_path, dst_label_path)
                
                src_image_path = os.path.join(images_dir, image_file)
                dst_image_path = os.path.join(output_dir, 'train', 'images', image_file)
                shutil.copy(src_image_path, dst_image_path)

            for file in val_files:
                image_file = file.replace('.txt', '.jpg')

                src_label_path = os.path.join(labels_dir, file)
                dst_label_path = os.path.join(output_dir, 'val', 'labels', file)
                shutil.copy(src_label_path, dst_label_path)

                src_image_path = os.path.join(images_dir, image_file)
                dst_image_path = os.path.join(output_dir, 'val', 'images', image_file)
                shutil.copy(src_image_path, dst_image_path)

    print("随机样本已收集完成")

# 示例用法
root_directory1 = '/workspace/data/飞行物体(>0.5%)'
root_directory2 = '/workspace/data/飞行物体(0.5%)'
root_directory3 = '/workspace/data/飞行物体(0.01%)'
output_directory1 = '/workspace/自建数据集/data(>0.5%)'
output_directory2 = '/workspace/自建数据集/data(0.5%)'
output_directory3 = '/workspace/自建数据集/data(0.01%)'
num_samples1 = 6000  # 指定要随机挑选的图片张数
num_samples2 = 6000
num_samples3 = 2000


collect_random_samples(root_directory1, output_directory1, num_samples1)
collect_random_samples(root_directory2, output_directory2, num_samples2)
collect_random_samples(root_directory3, output_directory3, num_samples3)

# root@c1249659bba3:/workspace# /usr/bin/python /workspace/飞行检测/数据挑选.py
# Found 30979 txt files in Drone(big): picking 3000 files
# Found 200 txt files in bird(middle): picking 200 files
# Found 4084 txt files in 0904-bird-AUG: picking 3000 files
# Found 6317 txt files in evtol: picking 3000 files
# Found 1846 txt files in V_PLANE(big): picking 1846 files
# Found 3251 txt files in V_HELICOPTER(big): picking 3000 files
# 随机样本已收集完成
# Found 3512 txt files in V_HELICOPTER(small): picking 3000 files
# Found 35909 txt files in Drone(small): picking 3000 files
# Found 4639 txt files in V_PLANE(small): picking 3000 files
# Found 1121 txt files in evtol(small): picking 1121 files
# Found 6990 txt files in bird(small): picking 3000 files
# 随机样本已收集完成
# Found 868 txt files in dots: picking 868 files
# Found 0 txt files in V_HELICOPTER(tiny): picking 0 files
# Found 4404 txt files in TYJT-0830-AUG: picking 3000 files
# Found 0 txt files in V_PLANE(tiny): picking 0 files
# Found 3580 txt files in Drone(tiny): picking 3000 files
# Found 168 txt files in bird(tiny): picking 168 files