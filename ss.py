import requests
import zmq
import torch
import threading
import random
import os
from model import ModelA  # 驾驶意图预测 LSTM
from model import ModelB  # 物体检测 FasterRCNN
from model import ModelC  # 车道线检测 UNet
from utils import decide_model_for_next_training

class ModelManager:
    def __init__(self):
        self.models = {
            'A': ModelA(),
            'B': ModelB(),
            'C': ModelC()
        }
        self.model_weights = {
            'A': self.models['A'].state_dict(),
            'B': self.models['B'].state_dict(),
            'C': self.models['C'].state_dict()
        }
        self.lock = threading.Lock()
        os.makedirs("server_logs", exist_ok=True)  # Ensure server_logs folder exists

        # Clear the log file at the start
        log_file_path = "server_logs/training_history.txt"
        with open(log_file_path, "w") as f:
            f.write("")  # Clear the contents of the file

    def aggregate_weights(self, model_name, client_weights):
        with self.lock:
            current_weights = self.model_weights[model_name]

            # Initialize an empty dictionary to store updated weights
            aggregated_weights = {}

            for key in current_weights.keys():
                # Convert weights to float for arithmetic operations
                client_weight = client_weights[key].float()  # Ensure client weight is float
                current_weight = current_weights[key].float()  # Ensure current weight is float
                
                # Perform aggregation (e.g., averaging)
                aggregated_weights[key] = (current_weight + client_weight) / 2

            # Update the model weights with the aggregated weights
            self.model_weights[model_name] = aggregated_weights


    def get_weights(self, model_name):
        with self.lock:
            return self.model_weights[model_name]

class ClientHandler(threading.Thread):
    def __init__(self, client_id, address, model_manager):
        super().__init__()
        self.client_id = client_id
        self.address = address
        self.model_manager = model_manager
        self.context = zmq.Context()
        
        # Set up the receiver socket
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind(f"tcp://*:{5555 + client_id}")
        
        # Set up the sender socket
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.connect(f"tcp://{address}:{6666 + client_id}")

    def run(self):
        while True:

            # Receive client weights
            msg = self.receiver.recv_pyobj()
            current_model = msg['model_name']
            weights = msg['weights']
            performance = msg['performance'] # Dict
            print(f"Received weights from {self.address}, model is {current_model}")
            print(performance)

            log_file_path = "server_logs/training_history.txt"
            with open(log_file_path, "a") as f:
                f.write(f"Client: {performance['client_id']}, "
                    f"Epoch: {performance['epoch'] + 1}, "
                    f"Model: {performance['model_name']}, "
                    f"Loss: {performance['epoch_loss']}, "
                    f"Epoch_duration: {performance['epoch_duration']}\n")
            # Aggregate weights
            self.model_manager.aggregate_weights(current_model, weights)

            
            performance['client_id'] = self.client_id
            
            # Randomly select the next model
            next_model = decide_model_for_next_training(performance)
            # print(f"Decision made by: {decision_source}")

            if next_model is None:
                next_model = random.choice(['A', 'B', 'C'])
            
            # Send weights after selecting next model
            aggregated_weights = self.model_manager.get_weights(next_model)
            self.sender.send_pyobj({
                'weights': aggregated_weights,
                'next_model': next_model
            })
            print(f"send weights to {self.address}, next model is {next_model}")


class ParameterServer:
    def __init__(self):
        self.model_manager = ModelManager()
        self.client_addresses = {
            # 1: "10.0.0.2",
            1: "10.0.0.15",
            2: "10.0.0.14",
            3: "10.0.0.90"
        }
        os.makedirs("server_logs", exist_ok=True) # Ensure server_logs folder exists

    def start(self):
        threads = []
        for client_id, address in self.client_addresses.items():
            thread = ClientHandler(client_id, address, self.model_manager)
            thread.start()
            threads.append(thread)
        
        print("Parameter server is running...")
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    server = ParameterServer()
    server.start()
