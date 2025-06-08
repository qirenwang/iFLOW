import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, Dataset
import random
from torchvision.datasets import CocoDetection
from PIL import Image
import os
import numpy as np
import json

# 设置固定的随机种子以确保初始化的一致性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Model A: Driving Intent Prediction (LSTM)
class ModelA(nn.Module):
    def __init__(self, input_size=2, hidden_size=45, num_classes=5):
        # 设置固定种子以确保初始化一致
        set_seed(1001)
        
        super(ModelA, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        # 使用固定的参数初始化
        self._init_weights()
        
    def _init_weights(self):
        # 为LSTM设置固定参数
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
        # 为全连接层设置固定参数
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.softmax(out)
        return out

# Model B: Object Detection (FasterRCNN)
class ModelB(nn.Module):
    def __init__(self):
        # 设置固定种子以确保初始化一致
        set_seed(1002)
        
        super(ModelB, self).__init__()
        # 使用预训练模型，确保权重加载的一致性
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
        
        # Modify to detect only two classes (background + 2 classes)
        num_classes = 3  # background + car + person
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    def _save_initial_weights(self):
        # 此函数只需运行一次来生成固定的初始权重
        torch.save(self.model.state_dict(), 'initial_weights_modelB.pth')
        
    def _load_initial_weights(self):
        # 如果有预保存的权重，则从文件加载以确保一致性
        if os.path.exists('initial_weights_modelB.pth'):
            self.model.load_state_dict(torch.load('initial_weights_modelB.pth'))
    
    def forward(self, x, targets=None):
        return self.model(x, targets)

# Model C: Lane Detection (UNet)
class ModelC(nn.Module):
    def __init__(self):
        # 设置固定种子以确保初始化一致
        set_seed(1003)
        
        super(ModelC, self).__init__()
        self.conv1 = DoubleConv(3, 32)
        self.conv2 = DoubleConv(32, 64)
        self.conv3 = DoubleConv(64, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(64, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        
        # 使用固定的参数初始化
        self._init_weights()
        
    def _init_weights(self):
        # 为每个卷积层设置固定初始参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.pool(conv1)
        conv2 = self.conv2(x)
        x = self.pool(conv2)
        x = self.conv3(x)
        x = self.upconv2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv6(x)
        x = self.upconv1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv7(x)
        return torch.sigmoid(self.final_conv(x))

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

def collate_fn(batch):
    return tuple(zip(*batch))

# Dataset classes
class DrivingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# BDD100K dataset class
class BDD100KDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = os.listdir(img_dir)
        
        # Map of category names to indices - only two classes
        self.category_map = {
            'car': 1,
            'person': 2
        }
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Create empty target with dummy boxes
        target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros(0, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }
        
        return image, target

class LaneDataset(Dataset):
    def __init__(self, frames_dir, masks_dir, transform=None, image_list=None):
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        if image_list is not None:
            self.images = image_list
        else:
            self.images = os.listdir(frames_dir)
            if len(self.images) > 192:
                self.images = random.sample(self.images, 192)

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

# Data loading functions
def get_driving_data(client_id):
    trainset = np.load('/home/orin/mmfl/datasets/brain4cars/trainset.npy')
    trainsety = np.load('/home/orin/mmfl/datasets/brain4cars/trainsety.npy')
    trainset = np.reshape(trainset, (43130, 90, 2))
    trainsety = np.reshape(trainsety, (43130, 1, 1))
    newtrainsety = np.zeros((43130, 5))
    for i in range(43130):
        val1 = int(trainsety[i, 0, 0])
        newtrainsety[i, val1] = 1
    
    # Sequential partitioning for driving dataset
    samples_per_client = 14376  # As specified
    start_idx = (client_id - 1) * samples_per_client
    end_idx = start_idx + samples_per_client
    # Ensure we don't go beyond dataset bounds
    if end_idx > len(trainset):
        end_idx = len(trainset)
    
    return DrivingDataset(trainset[start_idx:end_idx], newtrainsety[start_idx:end_idx])

def get_bdd100k_data(client_id):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((320, 320), antialias=True)
    ])
    
    # Correct paths to BDD100K dataset
    bdd_images_dir = '/home/orin/mmfl/datasets/100k/train'
    bdd_labels_dir = '/home/orin/mmfl/datasets/100k/annotations/100k/train'
    
    # Create dataset with images only - we'll handle labels in the training loop
    dataset = BDD100KDataset(
        img_dir=bdd_images_dir,
        transform=transform
    )
    
    # Sequential partitioning for BDD100K dataset
    images_per_client = 124  # Same as previously
    total_images = len(dataset)
    start_idx = (client_id - 1) * images_per_client
    end_idx = start_idx + images_per_client
    
    if end_idx > total_images:
        end_idx = total_images
    
    indices = list(range(start_idx, end_idx))
    subset = Subset(dataset, indices)
    
    return DataLoader(subset, batch_size=4, shuffle=True, collate_fn=collate_fn)

def get_lane_data(client_id):
    transform = T.Compose([T.Resize((256, 512)), T.ToTensor()])
    
    # Get all available image files
    frames_dir = '/home/orin/mmfl/datasets/tusimple/training/frames'
    all_images = sorted(os.listdir(frames_dir))  # Sort to ensure consistent ordering
    
    # Sequential partitioning for lane dataset
    images_per_client = 176  # As specified
    start_idx = (client_id - 1) * images_per_client
    end_idx = start_idx + images_per_client
    # Ensure we don't go beyond dataset bounds
    if end_idx > len(all_images):
        end_idx = len(all_images)
    
    # Select sequential subset based on client_id
    selected_images = all_images[start_idx:end_idx]
    
    return LaneDataset(
        frames_dir=frames_dir,
        masks_dir='/home/orin/mmfl/datasets/tusimple/training/lane-masks',
        transform=transform,
        image_list=selected_images
    )

def get_data_loader(model_name, client_id):
    if model_name == 'A':
        dataset = get_driving_data(client_id)
        return DataLoader(dataset, batch_size=4, shuffle=True)
    elif model_name == 'B':
        return get_bdd100k_data(client_id)  # Returns a configured DataLoader for BDD100K
    else:  # model_name == 'C'
        dataset = get_lane_data(client_id)
        return DataLoader(dataset, batch_size=4, shuffle=True)

# 更高级的方法：保存模型的固定初始参数，供两个框架使用
def save_initial_model_weights():
    """保存所有模型的初始权重，仅需运行一次"""
    # 确保权重目录存在
    os.makedirs('initial_weights', exist_ok=True)
    
    # 初始化并保存模型A的权重
    set_seed(1001)
    model_a = ModelA()
    torch.save(model_a.state_dict(), 'initial_weights/model_a_initial.pth')
    
    # 初始化并保存模型B的权重
    set_seed(1002)
    model_b = ModelB()
    torch.save(model_b.model.state_dict(), 'initial_weights/model_b_initial.pth')
    
    # 初始化并保存模型C的权重
    set_seed(1003)
    model_c = ModelC()
    torch.save(model_c.state_dict(), 'initial_weights/model_c_initial.pth')
    
    print("所有模型的初始权重已保存")

# 修改模型类的初始化函数以加载保存的权重
def load_saved_initial_weights():
    """从保存的文件加载初始权重"""
    if os.path.exists('initial_weights/model_a_initial.pth'):
        model_a = ModelA()
        model_a.load_state_dict(torch.load('initial_weights/model_a_initial.pth'))
        
        model_b = ModelB()
        model_b.model.load_state_dict(torch.load('initial_weights/model_b_initial.pth'))
        
        model_c = ModelC()
        model_c.load_state_dict(torch.load('initial_weights/model_c_initial.pth'))
        
        return model_a, model_b, model_c
    else:
        print("未找到保存的初始权重文件")
        return None, None, None

################################ 随机版 #######################################
# # model.py
# import torch
# import torch.nn as nn
# from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
# from torchvision.models.detection.faster_rcnn import FasterRCNN_MobileNet_V3_Large_FPN_Weights
# import torchvision.transforms as T
# from torch.utils.data import DataLoader, Subset, Dataset
# import random
# from torchvision.datasets import CocoDetection
# from PIL import Image
# import os
# import numpy as np

# # Model A: Driving Intent Prediction (LSTM)
# class ModelA(nn.Module):
#     def __init__(self, input_size=2, hidden_size=45, num_classes=5):
#         super(ModelA, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.3)
#         self.dropout = nn.Dropout(0.4)
#         self.fc = nn.Linear(hidden_size, num_classes)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         lstm_out = lstm_out[:, -1, :]
#         out = self.dropout(lstm_out)
#         out = self.fc(out)
#         out = self.softmax(out)
#         return out

# # Model B: Object Detection (FasterRCNN)
# class ModelB(nn.Module):
#     def __init__(self):
#         super(ModelB, self).__init__()
#         self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
    
#     def forward(self, x, targets=None):
#         return self.model(x, targets)

# # Model C: Lane Detection (UNet)
# class ModelC(nn.Module):
#     def __init__(self):
#         super(ModelC, self).__init__()
#         self.conv1 = DoubleConv(3, 32)
#         self.conv2 = DoubleConv(32, 64)
#         self.conv3 = DoubleConv(64, 128)
#         self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.conv6 = DoubleConv(128, 64)
#         self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
#         self.conv7 = DoubleConv(64, 32)
#         self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
#         self.pool = nn.MaxPool2d(2)

#     def forward(self, x):
#         conv1 = self.conv1(x)
#         x = self.pool(conv1)
#         conv2 = self.conv2(x)
#         x = self.pool(conv2)
#         x = self.conv3(x)
#         x = self.upconv2(x)
#         x = torch.cat([x, conv2], dim=1)
#         x = self.conv6(x)
#         x = self.upconv1(x)
#         x = torch.cat([x, conv1], dim=1)
#         x = self.conv7(x)
#         return torch.sigmoid(self.final_conv(x))

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# def collate_fn(batch):
#     return tuple(zip(*batch))

# # Dataset classes
# class DrivingDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = torch.FloatTensor(features)
#         self.labels = torch.FloatTensor(labels)
        
#     def __len__(self):
#         return len(self.features)
    
#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# class LaneDataset(Dataset):
#     def __init__(self, frames_dir, masks_dir, transform=None):
#         self.frames_dir = frames_dir
#         self.masks_dir = masks_dir
#         self.transform = transform
#         self.images = os.listdir(frames_dir)
#         if len(self.images) > 192:
#             self.images = random.sample(self.images, 192)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         img_path = os.path.join(self.frames_dir, img_name)
#         mask_path = os.path.join(self.masks_dir, img_name)
#         image = Image.open(img_path).convert('RGB')
#         mask = Image.open(mask_path).convert('L')
#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)
#         return image, mask

# # Data loading functions
# def get_driving_data(client_id):
#     trainset = np.load('/home/orin/mmfl/datasets/brain4cars/trainset.npy')
#     trainsety = np.load('/home/orin/mmfl/datasets/brain4cars/trainsety.npy')
#     trainset = np.reshape(trainset, (43130, 90, 2))
#     trainsety = np.reshape(trainsety, (43130, 1, 1))
#     newtrainsety = np.zeros((43130, 5))
#     for i in range(43130):
#         val1 = int(trainsety[i, 0, 0])
#         newtrainsety[i, val1] = 1
    
#     # Sequential partitioning for driving dataset
#     samples_per_client = 14376  # As specified
#     start_idx = (client_id - 1) * samples_per_client
#     end_idx = start_idx + samples_per_client
#     # Ensure we don't go beyond dataset bounds
#     if end_idx > len(trainset):
#         end_idx = len(trainset)
    
#     return DrivingDataset(trainset[start_idx:end_idx], newtrainsety[start_idx:end_idx])

# def get_coco_data(client_id):
#     transform = T.Compose([
#         T.ToTensor(),
#         T.Resize((320, 320), antialias=True)
#     ])
#     dataset = CocoDetection(
#         root='datasets/coco/train2017',
#         annFile='datasets/coco/annotations/instances_train2017.json',
#         transform=transform
#     )
    
#     # Sequential partitioning for COCO dataset
#     images_per_client = 192  # As specified
#     total_images = len(dataset)
#     start_idx = (client_id - 1) * images_per_client
#     end_idx = start_idx + images_per_client
#     # Ensure we don't go beyond dataset bounds
#     if end_idx > total_images:
#         end_idx = total_images
    
#     # Generate sequential indices
#     indices = list(range(start_idx, end_idx))
#     subset = Subset(dataset, indices)
#     return DataLoader(subset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# def get_lane_data(client_id):
#     transform = T.Compose([T.Resize((256, 512)), T.ToTensor()])
    
#     # Get all available image files
#     frames_dir = '/home/orin/mmfl/datasets/tusimple/training/frames'
#     all_images = sorted(os.listdir(frames_dir))  # Sort to ensure consistent ordering
    
#     # Sequential partitioning for lane dataset
#     images_per_client = 192  # As specified
#     start_idx = (client_id - 1) * images_per_client
#     end_idx = start_idx + images_per_client
#     # Ensure we don't go beyond dataset bounds
#     if end_idx > len(all_images):
#         end_idx = len(all_images)
    
#     # Select sequential subset based on client_id
#     selected_images = all_images[start_idx:end_idx]
    
#     return LaneDataset(
#         frames_dir=frames_dir,
#         masks_dir='/home/orin/mmfl/datasets/tusimple/training/lane-masks',
#         transform=transform,
#         image_list=selected_images
#     )

# def get_data_loader(model_name):
#     if model_name == 'A':
#         dataset = get_driving_data()
#         return DataLoader(dataset, batch_size=4, shuffle=True)
#     elif model_name == 'B':
#         return get_coco_data()  # 这里直接返回配置好的DataLoader
#     else:  # model_name == 'C'
#         dataset = get_lane_data()
#         return DataLoader(dataset, batch_size=4, shuffle=True)

############################ 备份 model.py #################################
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import torchvision
# from torchvision.models.detection import (
#     ssdlite320_mobilenet_v3_large,
#     ssd300_vgg16,
#     fasterrcnn_mobilenet_v3_large_fpn,
#     fasterrcnn_resnet50_fpn_v2,
# )
# from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
# from torchvision.models.detection.ssd import SSD300_VGG16_Weights
# from torchvision.models.detection.faster_rcnn import  FasterRCNN_MobileNet_V3_Large_FPN_Weights
# from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
# from torch.utils.data import DataLoader, Subset
# import torchvision.transforms as T
# import torch.optim as optim
# import random
# from torchvision.datasets import VOCDetection


# # Helper function to collate data batches
# def collate_fn(batch):
#     return tuple(zip(*batch))





# class ModelSSDlite(nn.Module):
#     def __init__(self):
#         super(ModelSSDlite, self).__init__()
#         # Load pre-trained SSDLite model with MobileNetV3 backbone
#         self.model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
#         # self.model.eval()  # Set the model to evaluation mode
    
#     def forward(self, x, targets=None):
#         # If targets are provided, calculate loss
#         if targets is not None:
#             return self.model(x, targets)
#         else:
#             return self.model(x)
    
# class ModelFasterRCNNMobileNet(nn.Module):
#     def __init__(self):
#         super(ModelFasterRCNNMobileNet, self).__init__()
#         # Load pre-trained Faster R-CNN model with MobileNetV3 backbone
#         self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
    
#     def forward(self, x, targets=None):
#         # If targets are provided, calculate loss
#         if targets is not None:
#             return self.model(x, targets)
#         else:
#             return self.model(x)
    
# class ModelFasterRCNNResNet(nn.Module):
#     def __init__(self):
#         super(ModelFasterRCNNResNet, self).__init__()
#         # Load pre-trained Faster R-CNN model with ResNet50 backbone
#         self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

    
#     def forward(self, x, targets=None):
#         # If targets are provided, calculate loss
#         if targets is not None:
#             return self.model(x, targets)
#         else:
#             return self.model(x)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 加载内建的COCO数据集
# def get_coco_dataset(train=True, subset_size=1000):
#     # COCO数据集的预处理
#     transforms = T.Compose([
#         T.ToTensor(),
#         T.Resize((320, 320)),  # SSDLite320需要的输入大小
#     ])
    
#     dataset = torchvision.datasets.CocoDetection(
#         root='datasets/coco/train2017' if train else 'datasets/coco/val2017',
#         annFile='datasets/coco/annotations/instances_train2017.json' if train else 'datasets/coco/annotations/instances_val2017.json',
#         transform=transforms
#     )
    
#     # 使用 Subset 选择数据集的一部分
#     if subset_size and subset_size < len(dataset):
#         indices = random.sample(range(len(dataset)), subset_size)
#         dataset = Subset(dataset, indices)
#     return dataset

# train_dataset = get_coco_dataset(train=True, subset_size=128)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)