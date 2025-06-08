import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from tqdm import tqdm
import time

# 定义一个简单的双卷积块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# 定义轻量级U-Net模型
class LightLaneNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.conv1 = DoubleConv(3, 32)
        self.conv2 = DoubleConv(32, 64)
        self.conv3 = DoubleConv(64, 128)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(64, 32)
        
        # Final layer
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        x = self.pool(conv1)
        
        conv2 = self.conv2(x)
        x = self.pool(conv2)
        
        x = self.conv3(x)
        
        # Decoder
        x = self.upconv2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv6(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv7(x)
        
        return torch.sigmoid(self.final_conv(x))

# 自定义数据集
class LaneDataset(Dataset):
    def __init__(self, frames_dir, masks_dir, transform=None):
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(frames_dir)
        
        # 随机选择128张图片
        if len(self.images) > 128:
            self.images = random.sample(self.images, 128)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.frames_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算并更新当前的平均loss
            current_loss = running_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}'
            })

# 主函数
def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 设置设备
    # device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集
    dataset = LaneDataset(
        frames_dir='/home/orin/mmfl/datasets/tusimple/training/frames',
        masks_dir='/home/orin/mmfl/datasets/tusimple/training/lane-masks',
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)} images")
    
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 初始化模型
    model = LightLaneNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    print("Starting training...")
    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device)
    
    # 保存模型
    model_save_path = '/home/orin/mmfl/model-pth-test/lane_detection_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == '__main__':
    main()