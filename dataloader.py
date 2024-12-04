import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from model import get_preprocessor_pipeline


class BirdDataset(Dataset):
    def __init__(self, data_csv_path, class_mapping_csv_path, root_dir, transform=None):
        """
        Args:
            data_csv_path (str): 包含文件路径和标签的CSV文件路径
            class_mapping_csv_path (str): 包含类别映射的CSV文件路径
            root_dir (str): 数据集根目录
            transform (callable, optional): 对样本应用的变换
        """
        # 加载CSV文件
        self.data_csv = pd.read_csv(data_csv_path)
        self.class_mapping = pd.read_csv(class_mapping_csv_path)

        # 构建类别映射字典
        self.class_dict = dict(
            zip(self.class_mapping['original_label'], self.class_mapping['new_label']))

        self.root_dir = root_dir
        self.transform = transform

    def get_n_num_class(self):
        return len(self.class_dict)

    def get_class_image_num(self):
        return self.data_csv['labels'].value_counts()

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            sample = self.data_csv.iloc[idx]
        else:
            sample = self.data_csv.iloc[idx.item()]

        # 图像路径
        img_path = os.path.join(self.root_dir, sample['filepaths'])
        # 加载图像
        image = Image.open(img_path).convert("RGB")
        # 标签

        label = int(sample['labels'].replace('class_', ''))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整到固定大小
        transforms.ToTensor(),         # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # 标准化
    ])

    # 定义数据集和DataLoader
    bird_dataset = BirdDataset(
        data_csv_path='all_data.csv',
        class_mapping_csv_path='class_mapping.csv',
        root_dir='.',
        transform=get_preprocessor_pipeline()
    )
    dataloader = DataLoader(bird_dataset, batch_size=32,
                            shuffle=True, num_workers=4)

    print(f"Number of classes: {bird_dataset.get_n_num_class()}")
    print(f"Number of images per class: {bird_dataset.get_class_image_num()}")

    # 测试DataLoader
    for images, labels in dataloader:
        print(f"Images batch shape: {images.size()}")
        print(f"Labels batch: {labels}")
        break
