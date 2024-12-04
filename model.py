# using resnet50 as the model
import timm
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


def get_resnet_50(class_num):
    model = models.resnet50(pretrained=True)
    # 修改全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_num)

    # set grad
    for param in model.parameters():
        param.requires_grad = True

    return model

def get_model_vit_16_224(class_num):
    # load vit using timm
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # 修改全连接层
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, class_num)

    # set grad
    for param in model.parameters():
        param.requires_grad = True

    return model

def get_val_pipeline():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 调整到固定大小
        transforms.ToTensor(),         # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

def get_preprocessor_pipeline():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 调整到固定大小
        transforms.ToTensor(),         # 转换为Tensor
        # argumentation
        transforms.RandomHorizontalFlip(),
        # random rotation
        transforms.RandomRotation(30),
        # random scale
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        # random brightness
        transforms.ColorJitter(brightness=0.1),
        # # random contrast
        transforms.ColorJitter(contrast=0.1),
        # # random saturation
        transforms.ColorJitter(saturation=0.1),
        # # random hue
        transforms.ColorJitter(hue=0.1),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])