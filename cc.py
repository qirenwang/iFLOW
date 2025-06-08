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
        
        # self.current_model = random.choice(['A', 'B', 'C'])
        self.current_model = 'A'
        self.epoch = 0
        
        self.loss_file_path = "loss/loss.txt"
        self.performance_file_path = f"loss/performance_{self.client_id}.txt"
        os.makedirs("loss", exist_ok=True)
        with open(self.loss_file_path, "w") as f:
            f.write("")
        with open(self.performance_file_path, "w") as f:
            f.write("")

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
        avg_gpu_utilization = round(sum(gpu_utilizations) / len(gpu_utilizations), 1) if gpu_utilizations and len(gpu_utilizations) > 0 else 'N/A'
        epoch_duration = round(time.time() - start_time, 1)
        
        self.performance_data = {
            'avg_cpu_percent': avg_cpu_percent,
            'avg_memory_usage': avg_memory_usage,
            'avg_gpu_utilization': avg_gpu_utilization,
            'epoch_duration': epoch_duration,
            'epoch_loss': epoch_loss,
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
                    f"Epoch duration: {epoch_duration} seconds\n")
        
        color_print(f"Model {self.current_model} trained on Client {self.client_id}: Epoch {self.epoch} complete.", 'blue')
        color_print(f"Average CPU usage: {avg_cpu_percent}%", 'yellow')
        color_print(f"Average memory usage: {avg_memory_usage}%", 'yellow')
        if gpu_utilizations and len(gpu_utilizations) > 0:
            color_print(f"Average GPU utilization: {avg_gpu_utilization}%", 'yellow')
        color_print(f"Epoch duration: {epoch_duration} seconds", 'yellow')

    def run(self, epochs=1):
        for _ in range(30):
            # 使用当前模型对应的数据加载器
            self.train(self.data_loaders[self.current_model], epochs)
            
            weights = {k: v.cpu().float() for k, v in self.models[self.current_model].state_dict().items()}
            self.sender.send_pyobj({
                'model_name': self.current_model,
                'weights': weights,
                "performance": self.performance_data
            })
            print(f"Sent weights to server for model {self.current_model}")
            
            msg = self.receiver.recv_pyobj()
            new_weights = {k: v.to(self.device) for k, v in msg['weights'].items()}
            self.current_model = msg['next_model']
            print(f"Received new weights for model {msg['next_model']}")
            self.models[self.current_model].load_state_dict(new_weights)

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    client = FederatedClient(client_id)
    client.run(epochs=1)

# import time
# import psutil
# import zmq
# import torch
# import random
# from model import ModelA, ModelB, ModelC
# from model import get_data_loader
# from torch.optim import SGD
# from utils import color_print
# import os
# import sys

# class FederatedClient:
#     def __init__(self, client_id, server_address="10.0.0.2"):
#         self.client_id = client_id
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Initialize models
#         self.models = {
#             'A': ModelA().to(self.device),
#             'B': ModelB().to(self.device),
#             'C': ModelC().to(self.device)
#         }
        
#         # Define optimizers
#         self.optimizers = {
#             'A': SGD(self.models['A'].parameters(), lr=0.001),
#             'B': SGD(self.models['B'].parameters(), lr=0.01),
#             'C': SGD(self.models['C'].parameters(), lr=0.01)
#         }

#         self.data_loaders = {
#             'A': get_data_loader('A', client_id),
#             'B': get_data_loader('B', client_id),
#             'C': get_data_loader('C', client_id)
#         }

#         self.context = zmq.Context()
#         self.sender = self.context.socket(zmq.PUSH)
#         self.sender.connect(f"tcp://{server_address}:{5555 + client_id}")
        
#         self.receiver = self.context.socket(zmq.PULL)
#         self.receiver.bind(f"tcp://*:{6666 + client_id}")
        
#         self.current_model = random.choice(['A', 'B', 'C'])
#         # self.current_model = 'C'
#         self.epoch = 0
        
#         self.loss_file_path = "loss/loss.txt"
#         self.performance_file_path = f"loss/performance_{self.client_id}.txt"
#         os.makedirs("loss", exist_ok=True)
#         with open(self.loss_file_path, "w") as f:
#             f.write("")
#         with open(self.performance_file_path, "w") as f:
#             f.write("")

#     def train(self, data_loader, epochs):
#         model = self.models[self.current_model]
#         optimizer = self.optimizers[self.current_model]

#         cpu_percentages = []
#         memory_usages = []
#         gpu_utilizations = [] if torch.cuda.is_available() else None
#         start_time = time.time()
        
#         for epoch in range(epochs):
#             running_loss = 0.0
#             batch_count = 0
            
#             if self.current_model == 'A':  # 驾驶意图预测
#                 for i, (features, targets) in enumerate(data_loader):
#                     features = features.to(self.device)
#                     targets = targets.to(self.device)
                    
#                     optimizer.zero_grad()
#                     outputs = model(features)
#                     loss = torch.nn.CrossEntropyLoss()(outputs, targets)
#                     loss.backward()
#                     optimizer.step()
                    
#                     running_loss += loss.item()
#                     batch_count += 1
                    
#             elif self.current_model == 'B':  # 目标检测
#                 for i, (images, targets) in enumerate(data_loader):
#                     # 准备输入
#                     images = [image.to(self.device) for image in images]
                    
#                     # 准备输入和目标
#                     valid_images = []
#                     valid_targets = []

#                     for image, target in zip(images, targets):
#                         boxes = []
#                         labels = []

#                         # 从COCO格式提取边界框和标签
#                         for ann in target:
#                             if 'bbox' in ann and 'category_id' in ann:
#                                 bbox = ann['bbox']  # [x, y, width, height]
#                                 if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
#                                     if bbox[2] > 0 and bbox[3] > 0:  # 检查宽度和高度是否为正
#                                         boxes.append([
#                                             bbox[0],
#                                             bbox[1],
#                                             bbox[0] + bbox[2],
#                                             bbox[1] + bbox[3]
#                                         ])
#                                         labels.append(ann['category_id'])

#                         if len(boxes) > 0 and len(labels) > 0:
#                             valid_images.append(image.to(self.device))
#                             valid_targets.append({
#                                 'boxes': torch.FloatTensor(boxes).to(self.device),
#                                 'labels': torch.tensor(labels, dtype=torch.int64).to(self.device)
#                             })

#                     if len(valid_images) == 0 or len(valid_targets) == 0:
#                         print(f"Skipping batch {i} due to no valid samples")
#                         continue

#                     optimizer.zero_grad()
#                     loss_dict = model(valid_images, valid_targets)
#                     loss = sum(loss for loss in loss_dict.values())
#                     loss.backward()
#                     optimizer.step()
                    
#                     running_loss += loss.item()
#                     batch_count += 1
                    
#             else:  # Model C: 车道线检测
#                 for i, (images, masks) in enumerate(data_loader):
#                     images = images.to(self.device)
#                     masks = masks.to(self.device)
                    
#                     optimizer.zero_grad()
#                     outputs = model(images)
#                     loss = torch.nn.BCELoss()(outputs, masks)
#                     loss.backward()
#                     optimizer.step()
                    
#                     running_loss += loss.item()
#                     batch_count += 1

#             # 收集性能指标
#             cpu_percentages.append(psutil.cpu_percent())
#             memory_usages.append(psutil.virtual_memory().percent)
#             if torch.cuda.is_available() and torch.cuda.max_memory_allocated(self.device) > 0:    
#                 gpu_utilizations.append(torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated(self.device) * 100)

#             # 计算每个epoch的平均损失
#             epoch_loss = running_loss / batch_count if batch_count > 0 else 0
#             print(f"{self.current_model} Epoch {epoch + 1}, Loss: {epoch_loss}")
#             with open("loss/loss.txt", "a") as f:
#                 f.write(f"Epoch: {self.epoch + 1}, Model: {self.current_model}, Loss: {epoch_loss}\n")

#         # 计算平均性能指标
#         avg_cpu_percent = round(sum(cpu_percentages) / len(cpu_percentages), 1)
#         avg_memory_usage = round(sum(memory_usages) / len(memory_usages), 1)
#         avg_gpu_utilization = round(sum(gpu_utilizations) / len(gpu_utilizations), 1) if len(gpu_utilizations)>0 else 'N/A'
#         epoch_duration = round(time.time() - start_time, 1)
        
#         self.performance_data = {
#             'avg_cpu_percent': avg_cpu_percent,
#             'avg_memory_usage': avg_memory_usage,
#             'avg_gpu_utilization': avg_gpu_utilization,
#             'epoch_duration': epoch_duration,
#             'epoch_loss': epoch_loss,
#             'client_id': self.client_id,
#             'epoch': self.epoch,
#             'model_name': self.current_model
#         }
#         self.epoch += 1
        
#         with open(self.performance_file_path, "a") as f:
#             f.write(f"Epoch: {self.epoch}, "
#                     f"Average CPU usage: {avg_cpu_percent}%, "
#                     f"Average memory usage: {avg_memory_usage}%, "
#                     f"Average GPU utilization: {avg_gpu_utilization}%, "
#                     f"Epoch duration: {epoch_duration} seconds\n")
        
#         color_print(f"Model {self.current_model} trained on Client {self.client_id}: Epoch {self.epoch} complete.", 'blue')
#         color_print(f"Average CPU usage: {avg_cpu_percent}%", 'yellow')
#         color_print(f"Average memory usage: {avg_memory_usage}%", 'yellow')
#         if gpu_utilizations:
#             color_print(f"Average GPU utilization: {avg_gpu_utilization}%", 'yellow')
#         color_print(f"Epoch duration: {epoch_duration} seconds", 'yellow')

#     def run(self, epochs=1):
#         for _ in range(30):
#             # 使用当前模型对应的数据加载器
#             self.train(self.data_loaders[self.current_model], epochs)
            
#             weights = {k: v.cpu().float() for k, v in self.models[self.current_model].state_dict().items()}
#             self.sender.send_pyobj({
#                 'model_name': self.current_model,
#                 'weights': weights,
#                 "performance": self.performance_data
#             })
#             print(f"Sent weights to server for model {self.current_model}")
            
#             msg = self.receiver.recv_pyobj()
#             new_weights = {k: v.to(self.device) for k, v in msg['weights'].items()}
#             self.current_model = msg['next_model']
#             print(f"Received new weights for model {msg['next_model']}")
#             self.models[self.current_model].load_state_dict(new_weights)

# if __name__ == "__main__":
#     client_id = int(sys.argv[1])
#     client = FederatedClient(client_id)
#     client.run(epochs=1)

########################  Backup cc.py #########################    
# import time
# import psutil
# import zmq
# import torch
# import random
# from model import ModelSSDlite as ModelA
# from model import ModelFasterRCNNMobileNet as ModelB
# from model import ModelFasterRCNNResNet as ModelC
# from torch.optim import SGD
# from model import train_loader 
# from utils import color_print
# import os
# import sys


# class FederatedClient:
#     def __init__(self, client_id, server_address="10.0.0.2"):
#         self.client_id = client_id
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.device = torch.device("cpu")
        
        
#         # Initialize models
#         self.models = {
#             'A': ModelA().to(self.device),
#             'B': ModelB().to(self.device),
#             'C': ModelC().to(self.device)
#         }
        
#         # Define optimizers
#         self.optimizers = {
#             'A': SGD(self.models['A'].parameters(), lr=0.01),
#             'B': SGD(self.models['B'].parameters(), lr=0.01),
#             'C': SGD(self.models['C'].parameters(), lr=0.01)
#         }

#         self.context = zmq.Context()
#         self.sender = self.context.socket(zmq.PUSH)
#         self.sender.connect(f"tcp://{server_address}:{5555 + client_id}")
        
#         self.receiver = self.context.socket(zmq.PULL)
#         self.receiver.bind(f"tcp://*:{6666 + client_id}")
        
#         self.current_model = random.choice(['A', 'B', 'C'])
#         self.epoch = 0
#         # Clear loss.txt at the start of the program
#         self.loss_file_path = "loss/loss.txt"
#         self.performance_file_path = f"loss/performance_{self.client_id}.txt"
#         os.makedirs("loss", exist_ok=True)  # 确保 loss 文件夹存在
#         with open(self.loss_file_path, "w") as f:
#             f.write("")  # 清空文件内容
#         with open(self.performance_file_path, "w") as f:
#             f.write("")  # Clear the file at the start

    # def train(self, data_loader, epochs):
    #     model = self.models[self.current_model]
    #     optimizer = self.optimizers[self.current_model]

    #     cpu_percentages = []
    #     memory_usages = []
    #     gpu_utilizations = [] if torch.cuda.is_available() else None
    #     start_time = time.time()
        
    #     for epoch in range(epochs):
    #         running_loss = 0.0
    #         batch_count = 0
    #         # Training loop
    #         for i, (images, targets) in enumerate(data_loader):
    #             # Prepare inputs
    #             images = [image.to(self.device) for image in images]
                
    #             # Prepare inputs and targets
    #             valid_images = []
    #             valid_targets = []

    #             for image, target in zip(images, targets):
    #                 boxes = []
    #                 labels = []

    #                 # Extract boxes and labels from COCO format
    #                 for ann in target:
    #                     if 'bbox' in ann and 'category_id' in ann:
    #                         bbox = ann['bbox']  # [x, y, width, height]
    #                         # Validate bbox values
    #                         if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
    #                             if bbox[2] > 0 and bbox[3] > 0:  # Check if width and height are positive
    #                                 # Convert to [x1, y1, x2, y2] format
    #                                 boxes.append([
    #                                     bbox[0],
    #                                     bbox[1],
    #                                     bbox[0] + bbox[2],
    #                                     bbox[1] + bbox[3]
    #                                 ])
    #                                 labels.append(ann['category_id'])

    #                 # Only add if we have valid boxes and labels
    #                 if len(boxes) > 0 and len(labels) > 0:
    #                     valid_images.append(image.to(self.device))  # Add valid image
    #                     valid_targets.append({
    #                         'boxes': torch.FloatTensor(boxes).to(self.device),
    #                         'labels': torch.tensor(labels, dtype=torch.int64).to(self.device)
    #                     })

    #             # Skip this batch if no valid samples
    #             if len(valid_images) == 0 or len(valid_targets) == 0:
    #                 print(f"Skipping batch {i} due to no valid samples")
    #                 continue

    #             # Pass valid images and their corresponding targets to the model
    #             optimizer.zero_grad()
    #             loss_dict = model(valid_images, valid_targets)

                


    #             losses = sum(loss for loss in loss_dict.values())
    #             # losses = torch.tensor(random.uniform(0.8, 0.95), requires_grad=True)
    #             losses.backward()
    #             optimizer.step()
                
    #             running_loss += losses.item()

    #             cpu_percentages.append(psutil.cpu_percent())
    #             memory_usages.append(psutil.virtual_memory().percent)
    #             if torch.cuda.is_available() and torch.cuda.max_memory_allocated(self.device) > 0:    
    #                 gpu_utilizations.append(torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated(self.device) * 100)
    #         # Calculate and print average loss
    #         epoch_loss = running_loss / len(data_loader)
    #         print(f"{self.current_model} Epoch {epoch + 1}, Loss: {epoch_loss}")
    #         with open("loss/loss.txt", "a") as f:
    #             f.write(f"Epoch: {self.epoch + 1}, Model: {self.current_model}, Loss: {epoch_loss}\n")

    #     # avg_loss = running_loss / batch_count if batch_count > 0 else 0

    #     running_loss = 0.0
    #     # batch_count = 0

    #     avg_cpu_percent = round(sum(cpu_percentages) / len(cpu_percentages), 1)
    #     avg_memory_usage = round(sum(memory_usages) / len(memory_usages), 1)
    #     avg_gpu_utilization = round(sum(gpu_utilizations) / len(gpu_utilizations), 1) if len(gpu_utilizations)>0 else 'N/A'
    #     epoch_duration = round(time.time() - start_time, 1)
    #     self.performance_data = {
    #         'avg_cpu_percent': avg_cpu_percent,
    #         'avg_memory_usage': avg_memory_usage,
    #         'avg_gpu_utilization': avg_gpu_utilization,
    #         'epoch_duration': epoch_duration,
    #         'epoch_loss': epoch_loss,
    #         'client_id': self.client_id,
    #         'epoch':self.epoch,
    #         'model_name': self.current_model
    #     }
    #     self.epoch += 1  
    #     # Write the performance metrics to the file in one line
    #     with open(self.performance_file_path, "a") as f:
    #         f.write(f"Epoch: {self.epoch}, "
    #                 f"Average CPU usage: {avg_cpu_percent}%, "
    #                 f"Average memory usage: {avg_memory_usage}%, "
    #                 f"Average GPU utilization: {avg_gpu_utilization}%, "
    #                 f"Epoch duration: {epoch_duration} seconds\n")
        
    #     color_print(f"Model {self.current_model} trained on Client {self.client_id}: Epoch {self.epoch} complete.", 'blue')
    #     color_print(f"Average CPU usage: {avg_cpu_percent}%", 'yellow')
    #     color_print(f"Average memory usage: {avg_memory_usage}%", 'yellow')
    #     if gpu_utilizations:
    #         color_print(f"Average GPU utilization: {avg_gpu_utilization}%", 'yellow')
    #     color_print(f"Epoch duration: {epoch_duration} seconds", 'yellow')
    

#     def run(self, data_loader, epochs=1):
#         # while True:
#         for _ in range(30):
#             # Train the current model
#             self.train(data_loader, epochs)
            
#             # Send weights to the server
#             weights = {k: v.cpu().float() for k, v in self.models[self.current_model].state_dict().items()}
#             self.sender.send_pyobj({
#                 'model_name': self.current_model,
#                 'weights': weights,
#                 "performance": self.performance_data
#             })
#             print(f"Sent weights to server for model {self.current_model}")
            
#             # Receive weights and the next model from the server
#             msg = self.receiver.recv_pyobj()
#             new_weights = {k: v.to(self.device) for k, v in msg['weights'].items()}
#             self.current_model = msg['next_model']
#             print(f"Received new weights for model {msg['next_model']}")
#             self.models[self.current_model].load_state_dict(new_weights)
            
            

# if __name__ == "__main__":
#     client_id = int(sys.argv[1])
#     client = FederatedClient(client_id)
#     client.run(train_loader, epochs=1)
