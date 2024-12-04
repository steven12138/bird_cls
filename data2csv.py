import os
import pandas as pd

# 指定图片文件夹路径
photo_dir = "data"

# 初始化列表以保存文件路径和标签
filepaths = []
labels = []

# 遍历每个子文件夹，每个文件夹名作为标签
for species in os.listdir(photo_dir):
    species_dir = os.path.join(photo_dir, species)
    if os.path.isdir(species_dir):
        # 检查类别文件夹内的图片数量
        images = os.listdir(species_dir)
        if len(images) >= 100:  # 只保留包含至少 100 张图片的类别
            for img_file in images:
                img_path = os.path.join(species_dir, img_file)
                filepaths.append(img_path)
                labels.append(species)

# 创建 DataFrame
df = pd.DataFrame({
    'filepaths': filepaths,
    'labels': labels
})

# 将 DataFrame 保存为单个 CSV 文件
df.to_csv("all_data.csv", index=False)

print("所有数据已生成并保存为 all_data.csv")
