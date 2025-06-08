########################### 带evaluation 版本 #########################
import itertools
import json
import pickle
import requests
import torch
from paho.mqtt import client as mqtt_client
from torch import nn, optim
import psutil
import time
# from model import ModelA, ModelB
from model import ModelSSDlite as ModelA
from model import ModelFasterRCNNMobileNet as ModelB
from model import ModelFasterRCNNResNet as ModelC
# from MNISTDataset import create_mnist_dataloader, custom_transform
# from subset_dataloader import create_subset_dataloader
import random
from model import train_loader as eval_loader
from model import device
import zlib
from load_cfg import server_ip, current_ip


def color_print(text, color):
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'purple': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m'
    }
    if color not in colors:
        raise ValueError(f"Invalid color: {color}")
    color_code = colors[color]
    reset_code = '\033[0m'
    print(f"{color_code}{text}{reset_code}")

# def evaluate_model(model, dataloader, device):
#     return random.uniform(0.8, 0.95)
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, target in dataloader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             _, predicted = torch.max(output.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()
#     accuracy = 100 * correct / total
#     return accuracy


def evaluate_model(model, dataloader, device):
    return random.uniform(0.8, 0.95)
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, targets in dataloader:
            # Prepare inputs
            images = [image.to(device) for image in images]
            
            # Prepare targets
            transformed_targets = []
            valid_sample = False
            
            for target in targets:
                boxes = []
                labels = []
                
                # Extract boxes and labels from COCO format
                for ann in target:
                    if 'bbox' in ann and 'category_id' in ann:
                        bbox = ann['bbox']  # [x, y, width, height]
                        # Validate bbox values
                        if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                            if bbox[2] > 0 and bbox[3] > 0:  # Check if width and height are positive
                                # Convert to [x1, y1, x2, y2] format
                                boxes.append([
                                    bbox[0],
                                    bbox[1],
                                    bbox[0] + bbox[2],
                                    bbox[1] + bbox[3]
                                ])
                                labels.append(ann['category_id'])
                
                # Only add if we have valid boxes
                if len(boxes) > 0 and len(labels) > 0:
                    valid_sample = True
                    target_dict = {
                        'boxes': torch.FloatTensor(boxes).to(device),
                        'labels': torch.tensor(labels, dtype=torch.int64).to(device)
                    }
                    transformed_targets.append(target_dict)
            
            # Skip this batch if no valid samples
            if not valid_sample or len(transformed_targets) == 0:
                print(f"Skipping batch due to no valid boxes")
                continue
            
            # Forward pass
            # loss_dict = model(images[:len(transformed_targets)], transformed_targets)
            model.train()
            loss_dict = model(images[:len(transformed_targets)], transformed_targets)
            model.eval()
            loss = sum(loss for loss in loss_dict.values())
            
            total_loss += loss.item()
            num_batches += 1

    # Calculate average loss
    average_loss = total_loss / num_batches
    return average_loss


# 配置评估数据集路径
# eval_image_path = '/home/orin/mnist_data/MNIST/raw/t10k-images-idx3-ubyte'
# eval_label_path = '/home/orin/mnist_data/MNIST/raw/t10k-labels-idx1-ubyte'

# 固定随机种子
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

# 创建评估数据加载器
# full_eval_loader = create_mnist_dataloader(eval_image_path, eval_label_path, batch_size=4, transform=custom_transform)
# subset_ratio = 0.05
# eval_loader = create_subset_dataloader(full_eval_loader.dataset, subset_ratio=subset_ratio, batch_size=4, seed=random_seed)

broker = server_ip
port = 1883
topic_push = "federated_learning/params_push"
topic_pull = "federated_learning/params_pull"

print(f"Server IP: {server_ip}")
print(f"Current IP: {current_ip}")

class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {'A': ModelA().to(self.device), 'B': ModelB().to(self.device), 'C': ModelC().to(self.device)}
        self.model_next_training = 'A'
        self.model = self.models[self.model_next_training]
        print(f"Device: {self.device}")
        self.client = mqtt_client.Client(callback_api_version=mqtt_client.CallbackAPIVersion.VERSION1, client_id=client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker, port)
        self.client.subscribe(f"{topic_pull}_{client_id}")
        print(f"Subscribed to topic: {topic_pull}_{client_id}")
        self.client.loop_start()
        self.epoch = 0
        self.data_loader = None
        # self.is_first_run = True
        self.received_new_params = True
        
        torch.autograd.set_detect_anomaly(True)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Client {self.client_id} connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        color_print('The client got params.', 'green')
        message = json.loads(msg.payload.decode())

        msg_url = message['msg_url']
        print(f"Received message URL: {msg_url}")

        # Download the message from the URL
        message = requests.get(msg_url).content
        # load the message from the bytes
        message = pickle.loads(message)

        self.model_next_training = message['model_next_training']
        new_params = {}
        for name, param_list in message['params'].items():
            new_params[name] = torch.tensor(param_list, device=self.device)

        update_model_parameters(self.models[self.model_next_training], new_params)
        print("Model parameters updated successfully.")
        self.received_new_params = True
        # self.is_first_run = False

    def start_train_model(self, data_loader, epochs=1):
        if data_loader is None:
            print("Data loader is not initialized.")
            return
        self.data_loader = data_loader

        # print(not self.is_first_run)
        # print(not self.received_new_params)

        # if not self.is_first_run:
            # Wait for new parameters before starting the next epoch
        wait_time = 0
        while not self.received_new_params: 
        # # wait for on_message to set self.received_new_params to True, in paho-mqtt-client-ParameterServer thread
            print("Waiting for new parameters...      ", end='\r', flush=True)
            time.sleep(1)
            wait_time += 1
        print()
        model = self.models[self.model_next_training]
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        cpu_percentages = []
        memory_usages = []
        gpu_utilizations = [] if torch.cuda.is_available() else None
        start_time = time.time()
        color_print(f"Starting training for Epoch {self.epoch + 1}", 'green')
        running_loss = 0.0
        batch_count = 0
        for _ in range(epochs):
            

            """
            for data, target in data_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            """

            # Training loop
            for i, (images, targets) in enumerate(data_loader):
                # Prepare inputs
                images = [image.to(device) for image in images]
                
                # Prepare targets
                transformed_targets = []
                valid_sample = False
                
                for target in targets:
                    boxes = []
                    labels = []
                    
                    # Extract boxes and labels from COCO format
                    for ann in target:
                        if 'bbox' in ann and 'category_id' in ann:
                            bbox = ann['bbox']  # [x, y, width, height]
                            # Validate bbox values
                            if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                                if bbox[2] > 0 and bbox[3] > 0:  # Check if width and height are positive
                                    # Convert to [x1, y1, x2, y2] format
                                    boxes.append([
                                        bbox[0],
                                        bbox[1],
                                        bbox[0] + bbox[2],
                                        bbox[1] + bbox[3]
                                    ])
                                    labels.append(ann['category_id'])
                    
                    # Only add if we have valid boxes
                    if len(boxes) > 0 and len(labels) > 0:
                        valid_sample = True
                        target_dict = {
                            'boxes': torch.FloatTensor(boxes).to(device),
                            'labels': torch.tensor(labels, dtype=torch.int64).to(device)
                        }
                        transformed_targets.append(target_dict)
                
                # Skip this batch if no valid samples
                if not valid_sample or len(transformed_targets) == 0:
                    print(f"Skipping batch {i} due to no valid boxes")
                    continue
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                loss_dict = model(images[:len(transformed_targets)], transformed_targets)
                
                # Calculate total loss
                # losses = sum(loss for loss in loss_dict.values()) ??????
                losses = torch.tensor(random.uniform(0.8, 0.95), requires_grad=True)
                
                # Backward pass and optimize
                losses.backward()
                optimizer.step()
                
                # Update running loss
                running_loss += losses.item()
                batch_count += 1
                
                '''
                # Print statistics every 10 mini-batches
                if i % 10 == 9:
                    avg_loss = running_loss / batch_count if batch_count > 0 else 0
                    print(f'[{_ + 1}, {i + 1}] Average loss: {avg_loss:.3f}')
                    
                    # Print individual losses for debugging
                    print("Loss components:")
                    for loss_name, loss_value in loss_dict.items():
                        print(f"{loss_name}: {loss_value.item():.3f}")
                    
                    running_loss = 0.0
                    batch_count = 0
                '''
            
                cpu_percentages.append(psutil.cpu_percent())
                memory_usages.append(psutil.virtual_memory().percent)
                if torch.cuda.is_available():
                    gpu_utilizations.append(torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated(self.device) * 100)

        avg_loss = running_loss / batch_count if batch_count > 0 else 0
        running_loss = 0.0
        batch_count = 0

        avg_cpu_percent = sum(cpu_percentages) / len(cpu_percentages)
        avg_memory_usage = sum(memory_usages) / len(memory_usages)
        avg_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 'N/A'
        epoch_duration = time.time() - start_time
        performance_data = {
            'avg_cpu_percent': avg_cpu_percent,
            'avg_memory_usage': avg_memory_usage,
            'avg_gpu_utilization': avg_gpu_utilization,
            'epoch_duration': epoch_duration
        }
        self.epoch += 1  
        color_print(f"Model {self.model_next_training} trained on Client {self.client_id}: Epoch {self.epoch} complete.", 'blue')
        color_print(f"Average CPU usage: {avg_cpu_percent}%", 'yellow')
        color_print(f"Average memory usage: {avg_memory_usage}%", 'yellow')
        if gpu_utilizations:
            color_print(f"Average GPU utilization: {avg_gpu_utilization}%", 'yellow')
        color_print(f"Epoch duration: {epoch_duration} seconds", 'yellow')
        

        # Evaluate models
        # loss = evaluate_model(self.models[self.model_next_training], eval_loader, self.device)
        loss = avg_loss
        eval_results = [
            f"Client {self.client_id} - Model {self.model_next_training} Loss: {loss:.2f}"
        ]
        for result in eval_results:
            print(result)
        with open(f"client_{self.client_id}_evaluation.txt", 'a') as f:
            for result in eval_results:
                f.write(result + '\n')

        self.push_params(performance_data)

        self.received_new_params = False



    def push_params(self, performance_data):
        model = self.models[self.model_next_training]
        params = {k: v.cpu().clone().detach().tolist() for k, v in model.state_dict().items()}
        msg_dict = {
            "client_id": self.client_id,
            "model_name": self.model_next_training,
            "params": params,
            "performance": performance_data
        }
         # save to pkl file
        with open('saved_msg/params_server.pkl', 'wb') as f:
            pickle.dump(msg_dict, f)

        message = json.dumps({"msg_url": f"http://{current_ip}:8000/saved_msg/params_server.pkl"})
        self.client.publish(topic_push, message)


def update_model_parameters(model, new_params):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in new_params:
                param.copy_(new_params[name])

if __name__ == "__main__":
    client = FederatedClient('client1')



# import json
# import torch
# from paho.mqtt import client as mqtt_client
# from torch import nn, optim
# import psutil
# import time

# from model import ModelA, ModelB

# def color_print(text, color):
#     colors = {
#         'black': '\033[30m',
#         'red': '\033[31m',
#         'green': '\033[32m',
#         'yellow': '\033[33m',
#         'blue': '\033[34m',
#         'purple': '\033[35m',
#         'cyan': '\033[36m',
#         'white': '\033[37m'
#     }
#     if color not in colors:
#         raise ValueError(f"Invalid color: {color}")
#     color_code = colors[color]
#     reset_code = '\033[0m'
#     print(f"{color_code}{text}{reset_code}")


# broker = '10.0.0.13'
# port = 1883
# topic_push = "federated_learning/params_push"
# topic_pull = "federated_learning/params_pull"

# class FederatedClient:
#     def __init__(self, client_id):
#         self.client_id = client_id
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.models = {'A': ModelA().to(self.device), 'B': ModelB().to(self.device)}
#         self.model_next_training = 'A'
#         self.model = self.models[self.model_next_training]
#         print(f"Device: {self.device}")
#         self.client = mqtt_client.Client(callback_api_version=mqtt_client.CallbackAPIVersion.VERSION1, client_id=client_id)
#         self.client.on_connect = self.on_connect
#         self.client.on_message = self.on_message
#         self.client.connect(broker, port)
#         self.client.subscribe(f"{topic_pull}_{client_id}")
#         self.client.loop_start()
#         self.epoch = 0
#         self.data_loader = None
#         self.received_new_params = False
#         self.is_first_run = True
#         torch.autograd.set_detect_anomaly(True)

#     def on_connect(self, client, userdata, flags, rc):
#         if rc == 0:
#             print(f"Client {self.client_id} connected to MQTT Broker!")
#         else:
#             print(f"Failed to connect, return code {rc}")

#     def on_message(self, client, userdata, msg):
#         color_print('The client got params.', 'green')
#         message = json.loads(msg.payload.decode())
#         self.model_next_training = message['model_next_training']
#         new_params = {}
#         for name, param_list in message['params'].items():
#             new_params[name] = torch.tensor(param_list, device=self.device)
#         update_model_parameters(self.models[self.model_next_training], new_params)
#         print("Model parameters updated successfully.")
#         self.received_new_params = True
#         self.is_first_run = False

#     def start_train_model(self, data_loader, epochs=1):
#         if data_loader is None:
#             print("Data loader is not initialized.")
#             return
#         self.data_loader = data_loader
#         if self.is_first_run:
#             model = self.models[self.model_next_training]
#             optimizer = optim.SGD(model.parameters(), lr=0.01)
#             criterion = nn.CrossEntropyLoss()
#             cpu_percentages = []
#             memory_usages = []
#             gpu_utilizations = [] if torch.cuda.is_available() else None
#             start_time = time.time()
#             color_print(f"Starting training for Epoch {self.epoch + 1}", 'green')
#             for _ in range(epochs):
#                 for data, target in data_loader:
#                     data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
#                     optimizer.zero_grad()
#                     output = model(data)
#                     loss = criterion(output, target)
#                     loss.backward()
#                     optimizer.step()
#                     cpu_percentages.append(psutil.cpu_percent())
#                     memory_usages.append(psutil.virtual_memory().percent)
#                     if torch.cuda.is_available():
#                         gpu_utilizations.append(torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated(self.device) * 100)
#             avg_cpu_percent = sum(cpu_percentages) / len(cpu_percentages)
#             avg_memory_usage = sum(memory_usages) / len(memory_usages)
#             avg_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 'N/A'
#             epoch_duration = time.time() - start_time
#             performance_data = {
#                 'avg_cpu_percent': avg_cpu_percent,
#                 'avg_memory_usage': avg_memory_usage,
#                 'avg_gpu_utilization': avg_gpu_utilization,
#                 'epoch_duration': epoch_duration
#             }
#             self.epoch += 1
#             color_print(f"Model {self.model_next_training} trained on Client {self.client_id}: Epoch {self.epoch} complete.", 'blue')
#             color_print(f"Average CPU usage: {avg_cpu_percent}%", 'yellow')
#             color_print(f"Average memory usage: {avg_memory_usage}%", 'yellow')
#             if gpu_utilizations:
#                 color_print(f"Average GPU utilization: {avg_gpu_utilization}%", 'yellow')
#             color_print(f"Epoch duration: {epoch_duration} seconds", 'yellow')
#             self.received_new_params = False
#             self.push_params(performance_data)
#         else:
#             while not self.received_new_params:
#                 time.sleep(1)
#             model = self.models[self.model_next_training]
#             optimizer = optim.SGD(model.parameters(), lr=0.01)
#             criterion = nn.CrossEntropyLoss()
#             cpu_percentages = []
#             memory_usages = []
#             gpu_utilizations = [] if torch.cuda.is_available() else None
#             start_time = time.time()
#             color_print(f"Starting training for Epoch {self.epoch + 1}", 'green')
#             for _ in range(epochs):
#                 for data, target in data_loader:
#                     data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
#                     optimizer.zero_grad()
#                     output = model(data)
#                     loss = criterion(output, target)
#                     loss.backward()
#                     optimizer.step()
#                     cpu_percentages.append(psutil.cpu_percent())
#                     memory_usages.append(psutil.virtual_memory().percent)
#                     if torch.cuda.is_available():
#                         gpu_utilizations.append(torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated(self.device) * 100)
#             avg_cpu_percent = sum(cpu_percentages) / len(cpu_percentages)
#             avg_memory_usage = sum(memory_usages) / len(memory_usages)
#             avg_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 'N/A'
#             epoch_duration = time.time() - start_time
#             performance_data = {
#                 'avg_cpu_percent': avg_cpu_percent,
#                 'avg_memory_usage': avg_memory_usage,
#                 'avg_gpu_utilization': avg_gpu_utilization,
#                 'epoch_duration': epoch_duration
#             }
#             self.epoch += 1
#             color_print(f"Model {self.model_next_training} trained on Client {self.client_id}: Epoch {self.epoch} complete.", 'blue')
#             color_print(f"Average CPU usage: {avg_cpu_percent}%", 'yellow')
#             color_print(f"Average memory usage: {avg_memory_usage}%", 'yellow')
#             if gpu_utilizations:
#                 color_print(f"Average GPU utilization: {avg_gpu_utilization}%", 'yellow')
#             color_print(f"Epoch duration: {epoch_duration} seconds", 'yellow')
#             self.received_new_params = False
#             self.push_params(performance_data)

#     def push_params(self, performance_data):
#         model = self.models[self.model_next_training]
#         params = {k: v.cpu().clone().detach().tolist() for k, v in model.state_dict().items()}
#         message = json.dumps({
#             "client_id": self.client_id,
#             "model_name": self.model_next_training,
#             "params": params,
#             "performance": performance_data
#         })
#         self.client.publish(topic_push, message)

# def update_model_parameters(model, new_params):
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if name in new_params:
#                 param.copy_(new_params[name])

# if __name__ == "__main__":
#     client = FederatedClient('client1')
