import argparse
import glob
import shutil
import sys

import torch
from tqdm import tqdm

from dataloader import BirdDataset
from model import get_preprocessor_pipeline, get_resnet_50, get_val_pipeline, get_model_vit_16_224
from torch.utils.data import DataLoader

import tensorboardX

if __name__ == '__main__':

    # tensorboard
    lr = 0.0000000005
    optim = 'adam'

    print(f"Training with lr: {lr} and optim: {optim}")
    writer = tensorboardX.SummaryWriter(
        f"runs/lr_{lr}_optim_{optim}"
    )

    bird_dataset = BirdDataset(
        data_csv_path='all_data.csv',
        class_mapping_csv_path='class_mapping.csv',
        root_dir='.',
        transform=get_preprocessor_pipeline()
    )
    # split train / val
    train_size = int(0.9 * len(bird_dataset))
    val_size = len(bird_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(bird_dataset, [train_size, val_size])
    # val不做augmentation
    val_dataset.transform = get_val_pipeline()

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    n_num_class = bird_dataset.get_n_num_class()

    model = get_model_vit_16_224(n_num_class)

    # using cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = None
    if optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.9)

    # trying to resume training
    # check ckpt/model_epoch_*.pth exists

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, help='resume training from latest checkpoint', default=None)
    args = parser.parse_args()
    if args.ckpt is not None:
        # get latest checkpoint
        filename = args.ckpt
        checkpoint = torch.load(filename)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        # writer = checkpoint['writer']
        print(f"Resuming training from {filename}")

    for epoch in range(200):
        model.train()
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader)) as pbar:
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.5f}")
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + i)
                lr_scheduler.step()
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch * len(train_dataloader) + i)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, total=len(val_dataloader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch} Val Accuracy: {100 * correct / total}")
        writer.add_scalar('Accuracy/val', 100 * correct / total, epoch)

        # mkdir ckpt if not exists
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }, f"ckpt/vit_{lr}_{optim}_epoch_{epoch}_acc_{100 * correct / total:.2f}.pth")
    writer.close()
