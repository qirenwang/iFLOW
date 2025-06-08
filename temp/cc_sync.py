import time
import psutil
import zmq
import torch
import random
from model import ModelA, ModelB, ModelC
from model import get_data_loader
from torch.optim import SGD
from utils import color_print
import os
import sys

class FederatedClient:
    def __init__(self, client_id, server_address="10.0.0.2"):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.models = {
            'A': ModelA().to(self.device),
            'B': ModelB().to(self.device),
            'C': ModelC().to(self.device)
        }
        
        # Define optimizers
        self.optimizers = {
            'A': SGD(self.models['A'].parameters(), lr=0.001),
            'B': SGD(self.models['B'].parameters(), lr=0.0001),
            'C': SGD(self.models['C'].parameters(), lr=0.01)
        }

        self.data_loaders = {
            'A': get_data_loader('A', client_id),
            'B': get_data_loader('B', client_id),
            'C': get_data_loader('C', client_id)
        }

        self.context = zmq.Context()
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.connect(f"tcp://{server_address}:{5555 + client_id}")
        
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind(f"tcp://*:{6666 + client_id}")
        
        self.current_model = random.choice(['A', 'B', 'C'])
        # self.current_model = 'A'

        self.epoch = 0
        self.idle_time = 0  # Initialize idle time
        
        # Create directories and files for logging
        self.loss_file_path = "loss/loss.txt"
        self.performance_file_path = f"loss/performance_{self.client_id}.txt"
        self.idle_time_file_path = f"loss/idle_time_{self.client_id}.txt"
        os.makedirs("loss", exist_ok=True)
        
        # Clear log files at start
        with open(self.loss_file_path, "w") as f:
            f.write("")
        with open(self.performance_file_path, "w") as f:
            f.write("")
        with open(self.idle_time_file_path, "w") as f:
            f.write("Round, Idle Time (seconds)\n")

    def train(self, data_loader, epochs):
        model = self.models[self.current_model]
        optimizer = self.optimizers[self.current_model]

        cpu_percentages = []
        memory_usages = []
        gpu_utilizations = [] if torch.cuda.is_available() else None
        start_time = time.time()
        
        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0
            
            if self.current_model == 'A':  # 驾驶意图预测
                for i, (features, targets) in enumerate(data_loader):
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = torch.nn.CrossEntropyLoss()(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1
                    
            elif self.current_model == 'B':  # 目标检测 (BDD100K)
                for i, (images, targets) in enumerate(data_loader):
                    images = [image.to(self.device) for image in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Generate random boxes and labels for training - only two classes
                    for i in range(len(targets)):
                        # Create random boxes (between 1-5 boxes per image)
                        num_boxes = random.randint(1, 5)
                        random_boxes = []
                        random_labels = []
                        
                        for _ in range(num_boxes):
                            # Random box coordinates
                            x1 = random.uniform(0, 240)
                            y1 = random.uniform(0, 240)
                            w = random.uniform(30, 100)
                            h = random.uniform(30, 100)
                            x2 = min(x1 + w, 320)
                            y2 = min(y1 + h, 320)
                            
                            random_boxes.append([x1, y1, x2, y2])
                            # Only use class 1 (car) or 2 (person)
                            random_labels.append(random.choice([1, 2])) 
                        
                        # Update target with random boxes and labels
                        targets[i]['boxes'] = torch.tensor(random_boxes, dtype=torch.float32).to(self.device)
                        targets[i]['labels'] = torch.tensor(random_labels, dtype=torch.int64).to(self.device)
                    
                    optimizer.zero_grad()
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1
                    
            else:  # Model C: 车道线检测
                for i, (images, masks) in enumerate(data_loader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = torch.nn.BCELoss()(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1

            # 收集性能指标
            cpu_percentages.append(psutil.cpu_percent())
            memory_usages.append(psutil.virtual_memory().percent)
            if torch.cuda.is_available() and torch.cuda.max_memory_allocated(self.device) > 0:    
                gpu_utilizations.append(torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated(self.device) * 100)

            # 计算每个epoch的平均损失
            epoch_loss = running_loss / batch_count if batch_count > 0 else 0
            print(f"{self.current_model} Epoch {epoch + 1}, Loss: {epoch_loss}")
            with open("loss/loss.txt", "a") as f:
                f.write(f"Epoch: {self.epoch + 1}, Model: {self.current_model}, Loss: {epoch_loss}\n")

        # 计算平均性能指标
        avg_cpu_percent = round(sum(cpu_percentages) / len(cpu_percentages), 1)
        avg_memory_usage = round(sum(memory_usages) / len(memory_usages), 1)
        avg_gpu_utilization = round(sum(gpu_utilizations) / len(gpu_utilizations), 1) if gpu_utilizations and len(gpu_utilizations)>0 else 'N/A'
        epoch_duration = round(time.time() - start_time, 1)
        
        # 创建性能数据，显式包含idle_time
        self.performance_data = {
            'avg_cpu_percent': avg_cpu_percent,
            'avg_memory_usage': avg_memory_usage,
            'avg_gpu_utilization': avg_gpu_utilization,
            'epoch_duration': epoch_duration,
            'epoch_loss': epoch_loss,
            'idle_time': self.idle_time,  # 确保包含上一轮的idle time
            'client_id': self.client_id,
            'epoch': self.epoch,
            'model_name': self.current_model
        }
        self.epoch += 1
        
        with open(self.performance_file_path, "a") as f:
            f.write(f"Epoch: {self.epoch}, "
                    f"Average CPU usage: {avg_cpu_percent}%, "
                    f"Average memory usage: {avg_memory_usage}%, "
                    f"Average GPU utilization: {avg_gpu_utilization}%, "
                    f"Epoch duration: {epoch_duration} seconds, "
                    f"Last round idle time: {self.idle_time} seconds\n")
        
        color_print(f"Model {self.current_model} trained on Client {self.client_id}: Epoch {self.epoch} complete.", 'blue')
        color_print(f"Average CPU usage: {avg_cpu_percent}%", 'yellow')
        color_print(f"Average memory usage: {avg_memory_usage}%", 'yellow')
        if gpu_utilizations and len(gpu_utilizations) > 0:
            color_print(f"Average GPU utilization: {avg_gpu_utilization}%", 'yellow')
        color_print(f"Epoch duration: {epoch_duration} seconds", 'yellow')
        if self.idle_time > 0:
            color_print(f"Previous round idle time: {self.idle_time} seconds", 'red')

    def run(self, epochs=1):
        for round_idx in range(30):
            # 使用当前模型对应的数据加载器
            self.train(self.data_loaders[self.current_model], epochs)
            
            # 确认性能数据中包含了idle time
            print(f"Performance data before sending - idle_time: {self.performance_data['idle_time']}")
            
            # 发送权重和性能数据到服务器
            weights = {k: v.cpu().float() for k, v in self.models[self.current_model].state_dict().items()}
            self.sender.send_pyobj({
                'model_name': self.current_model,
                'weights': weights,
                'performance': self.performance_data
            })
            color_print(f"Sent weights to server for model {self.current_model} at {time.strftime('%H:%M:%S')}", 'green')
            
            # 开始测量idle time
            idle_start_time = time.time()
            
            # 等待服务器响应 (此时客户端处于闲置状态)
            print(f"Client {self.client_id} waiting for server response...")
            msg = self.receiver.recv_pyobj()
            
            # 计算idle time
            idle_end_time = time.time()
            self.idle_time = round(idle_end_time - idle_start_time, 1)
            
            # 记录idle time
            with open(self.idle_time_file_path, "a") as f:
                f.write(f"{round_idx + 1}, {self.idle_time}\n")
            
            color_print(f"Client {self.client_id} was idle for {self.idle_time} seconds waiting for synchronization", 'red')
            
            # 处理服务器响应
            new_weights = {k: v.to(self.device) for k, v in msg['weights'].items()}
            self.current_model = msg['next_model']
            color_print(f"Received new weights for model {msg['next_model']} at {time.strftime('%H:%M:%S')}", 'green')
            self.models[self.current_model].load_state_dict(new_weights)

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    client = FederatedClient(client_id)
    client.run(epochs=1)