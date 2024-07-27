import sys
import os
import torch
from FederatedClient import FederatedClient
from MNISTDataset import create_mnist_dataloader, custom_transform
from subset_dataloader import create_subset_dataloader

def save_weights(model, save_dir, client_id, model_label):
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    file_path = os.path.join(save_dir, f'{client_id}_{model_label}_weights.pth')
    torch.save(model.state_dict(), file_path)  # 保存模型的状态字典
    print(f'Saved weights to {file_path}')  # 打印权重保存的位置

def update_model_parameters(model, new_params):
    # 安全加载新参数，不影响梯度计算
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in new_params:
                print(f"Updating parameter {name}")
                param.copy_(new_params[name])

client_id = sys.argv[1] if len(sys.argv) > 1 else 'client1'

train_image_path = '/home/orin/mnist_data/MNIST/raw/train-images-idx3-ubyte'
train_label_path = '/home/orin/mnist_data/MNIST/raw/train-labels-idx1-ubyte'

# 创建数据加载器
full_loader = create_mnist_dataloader(train_image_path, train_label_path, batch_size=4, transform=custom_transform)
subset_ratio = 0.1
train_loader = create_subset_dataloader(full_loader.dataset, subset_ratio=subset_ratio, batch_size=4)

# 初始化联邦学习客户端
client = FederatedClient(client_id)
client.data_loader = train_loader

# 设置最大训练周期数
max_epochs = 10
current_epoch = 0

try:
    while current_epoch < max_epochs:
        print(f"Starting training for Epoch {current_epoch + 1}")
        client.start_train_model(client.data_loader, epochs=1)
        save_weights(client.models[client.model_next_training], 'weights', client_id, f'Model_{client.model_next_training}')
        print(f"Completed Epoch {current_epoch + 1}")
        current_epoch += 1  # 增加周期计数器
except KeyboardInterrupt:
    print("Client is shutting down.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
