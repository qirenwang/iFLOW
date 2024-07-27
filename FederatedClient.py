import json
import torch
from paho.mqtt import client as mqtt_client
from torch import nn, optim
import psutil
import time

from model import ModelA, ModelB

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

broker = '10.0.0.13'
port = 1883
topic_push = "federated_learning/params_push"
topic_pull = "federated_learning/params_pull"

class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {'A': ModelA().to(self.device), 'B': ModelB().to(self.device)}
        self.model_next_training = 'A'
        self.model = self.models[self.model_next_training]
        print(f"Device: {self.device}")
        self.client = mqtt_client.Client(callback_api_version=mqtt_client.CallbackAPIVersion.VERSION1, client_id=client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker, port)
        self.client.subscribe(f"{topic_pull}_{client_id}")
        self.client.loop_start()
        self.epoch = 0
        self.data_loader = None
        self.received_new_params = False
        self.is_first_run = True
        torch.autograd.set_detect_anomaly(True)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Client {self.client_id} connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        color_print('The client got params.', 'green')
        message = json.loads(msg.payload.decode())
        self.model_next_training = message['model_next_training']
        new_params = {}
        for name, param_list in message['params'].items():
            new_params[name] = torch.tensor(param_list, device=self.device)
        update_model_parameters(self.models[self.model_next_training], new_params)
        print("Model parameters updated successfully.")
        self.received_new_params = True
        self.is_first_run = False

    def start_train_model(self, data_loader, epochs=1):
        if data_loader is None:
            print("Data loader is not initialized.")
            return
        self.data_loader = data_loader
        if self.is_first_run:
            model = self.models[self.model_next_training]
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            cpu_percentages = []
            memory_usages = []
            gpu_utilizations = [] if torch.cuda.is_available() else None
            start_time = time.time()
            color_print(f"Starting training for Epoch {self.epoch + 1}", 'green')
            for _ in range(epochs):
                for data, target in data_loader:
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    cpu_percentages.append(psutil.cpu_percent())
                    memory_usages.append(psutil.virtual_memory().percent)
                    if torch.cuda.is_available():
                        gpu_utilizations.append(torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated(self.device) * 100)
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
            self.received_new_params = False
            self.push_params(performance_data)
        else:
            while not self.received_new_params:
                time.sleep(1)
            model = self.models[self.model_next_training]
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            cpu_percentages = []
            memory_usages = []
            gpu_utilizations = [] if torch.cuda.is_available() else None
            start_time = time.time()
            color_print(f"Starting training for Epoch {self.epoch + 1}", 'green')
            for _ in range(epochs):
                for data, target in data_loader:
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    cpu_percentages.append(psutil.cpu_percent())
                    memory_usages.append(psutil.virtual_memory().percent)
                    if torch.cuda.is_available():
                        gpu_utilizations.append(torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated(self.device) * 100)
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
            self.received_new_params = False
            self.push_params(performance_data)

    def push_params(self, performance_data):
        model = self.models[self.model_next_training]
        params = {k: v.cpu().clone().detach().tolist() for k, v in model.state_dict().items()}
        message = json.dumps({
            "client_id": self.client_id,
            "model_name": self.model_next_training,
            "params": params,
            "performance": performance_data
        })
        self.client.publish(topic_push, message)

def update_model_parameters(model, new_params):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in new_params:
                param.copy_(new_params[name])

if __name__ == "__main__":
    client = FederatedClient('client1')
