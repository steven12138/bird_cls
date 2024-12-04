import os
import pandas as pd

# 设置原始数据路径和目标路径
source_dir = "data_zhong"  # 原始数据文件夹路径
target_dir = "data"  # 重命名后保存的文件夹路径

# 创建目标文件夹（如果不存在）
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 创建映射关系列表
mapping = []

# 遍历类别文件夹，将中文类别文件夹重命名为编号
for idx, category in enumerate(os.listdir(source_dir)):
    category_path = os.path.join(source_dir, category)
    target_category_name = f"class_{idx}"
    target_category_path = os.path.join(target_dir, target_category_name)
    
    # 仅处理文件夹
    if os.path.isdir(category_path):
        # 创建目标类别文件夹
        os.makedirs(target_category_path, exist_ok=True)
        
        # 移动所有图片到新的文件夹
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            target_img_path = os.path.join(target_category_path, img_file)
            os.rename(img_path, target_img_path)  # 移动文件

        # 记录映射关系
        mapping.append({"original_label": category, "new_label": target_category_name, "class": idx})

print("所有文件已重命名并保存到 data_renamed 文件夹中。")

# 保存映射关系到 CSV 文件
mapping_df = pd.DataFrame(mapping)
mapping_df.to_csv("class_mapping.csv", index=False)
print("映射关系已保存到 class_mapping.csv 文件中。")

# 读取原始 CSV 文件并更新文件路径
original_csv_path = "all_data.csv"
df = pd.read_csv(original_csv_path)

# 更新文件路径
def update_path(row):
    original_label = row['labels']
    new_label = mapping_df[mapping_df['original_label'] == original_label]['new_label'].values[0]
    new_path = row['filepaths'].replace(f"data\\{original_label}", f"{target_dir}\\{new_label}")
    return new_path

df['filepaths'] = df.apply(update_path, axis=1)
df['labels'] = df['labels'].map(lambda x: mapping_df[mapping_df['original_label'] == x]['new_label'].values[0])

# 保存更新后的 CSV 文件
df.to_csv("all_data_updated.csv", index=False)
print("已更新文件路径并保存到 all_data_updated.csv 文件中。")
