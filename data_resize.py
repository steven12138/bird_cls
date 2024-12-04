import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 设置原始数据和目标文件夹路径
source_dir = "data2"  # 原始图片文件夹路径
target_dir = "data"  # 调整大小后保存的文件夹路径

# 目标尺寸
target_size = (512, 512)  # 修改为你需要的尺寸，如 (960, 960)

# 创建目标文件夹（如果不存在）
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 获取所有图片路径
all_images = []
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if os.path.isdir(category_path):
        # 创建对应的目标类别文件夹
        target_category_path = os.path.join(target_dir, category)
        if not os.path.exists(target_category_path):
            os.makedirs(target_category_path)
        
        # 获取该类别文件夹中的所有图片路径
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            # 将目标路径的扩展名强制设为 .jpg
            target_img_path = os.path.join(target_category_path, os.path.splitext(img_file)[0] + ".jpg")
            all_images.append((img_path, target_img_path))

# 定义图片处理函数：等比例缩放 + 居中裁剪
def process_image(paths):
    img_path, target_img_path = paths
    try:
        with Image.open(img_path) as img:
            # 检查并转换为 RGB 模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 计算等比例缩放尺寸
            img_ratio = img.width / img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # 宽度大于目标尺寸比例，按高度缩放
                new_height = target_size[1]
                new_width = int(new_height * img_ratio)
            else:
                # 高度大于目标尺寸比例，按宽度缩放
                new_width = target_size[0]
                new_height = int(new_width / img_ratio)
            
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 中心裁剪
            left = (new_width - target_size[0]) / 2
            top = (new_height - target_size[1]) / 2
            right = left + target_size[0]
            bottom = top + target_size[1]
            
            img_cropped = img_resized.crop((left, top, right, bottom))
            img_cropped.save(target_img_path, format='JPEG')  # 保存为 JPEG 格式
        return True
    except Exception as e:
        print(f"无法处理图片 {img_path}: {e}")
        return False

# 使用多线程处理图片并显示进度条
with ThreadPoolExecutor() as executor:
    # 提交任务
    futures = [executor.submit(process_image, img_paths) for img_paths in all_images]
    
    # 显示进度条
    for _ in tqdm(as_completed(futures), total=len(futures), desc="处理图片"):
        pass

print("所有图片已调整大小、居中裁剪，并保存为 JPG 格式到 photo_new 文件夹中。")
