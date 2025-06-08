import numpy as np
import requests
import zmq
import torch
import threading
import random
import os
import time  # 添加时间模块
from model import ModelA  # 驾驶意图预测 LSTM
from model import ModelB  # 物体检测 FasterRCNN
from model import ModelC  # 车道线检测 UNet
from utils import decide_model_for_next_training
from utils import color_print
from threading import Barrier
from Ranklist_Multi_UCB_Algo import UCBCalculator, ranklist_multi_ucb

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
        log_file_path = "server_logs/training_history_ranklist.txt"
        with open(log_file_path, "w") as f:
            f.write("")  # Clear the contents of the file

        self.performance_data = [None] * 3  # Shared list to store performance data from each client
        self.barrier = Barrier(3)  # Barrier for 3 clients + 1 main thread

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
            client_finish_time = time.time()
            current_model = msg['model_name']
            weights = msg['weights']
            performance = msg['performance']  # Dict
            
            print(f"Received weights from {self.address}, model is {current_model} at {time.strftime('%H:%M:%S')}")
            
            # 打印收到的idle time
            idle_time = performance.get('idle_time', 0)
            print(f"Client {self.client_id} reported idle time: {idle_time} seconds")
            
            log_file_path = "server_logs/training_history_ranklist.txt"
            with open(log_file_path, "a") as f:
                f.write(f"Client: {performance['client_id']}, "
                    f"Epoch: {performance['epoch'] + 1}, "
                    f"Model: {performance['model_name']}, "
                    f"Loss: {performance['epoch_loss']}, "
                    f"Epoch_duration: {performance['epoch_duration']}, "
                    f"Idle_time: {idle_time}\n")
            
            # Aggregate weights
            self.model_manager.aggregate_weights(current_model, weights)

            # Store performance data
            self.model_manager.performance_data[self.client_id - 1] = performance
            
            # 记录屏障等待开始时间
            barrier_start_time = time.time()
            print(f"Client {self.client_id} waiting at barrier at {time.strftime('%H:%M:%S')}")
            
            # Wait for all clients to reach this point
            self.model_manager.barrier.wait()
            
            # 计算屏障等待时间
            barrier_wait_time = time.time() - barrier_start_time
            print(f"Client {self.client_id} waited at barrier for {round(barrier_wait_time, 1)} seconds")

            # Main thread decides the next model for each client
            if self.client_id == 1:  # Only one thread should perform the decision
                decision_start_time = time.time()
                next_models = decide_model_for_next_training(self.model_manager.performance_data)
                self.model_manager.next_models = next_models
                decision_time = time.time() - decision_start_time
                print(f"Model decision took {round(decision_time, 1)} seconds")

            # Wait for the decision to be made
            self.model_manager.barrier.wait()

            # Get the next model for this client
            next_model = self.model_manager.next_models[self.client_id - 1]

            # Send weights after selecting next model
            aggregated_weights = self.model_manager.get_weights(next_model)
            self.sender.send_pyobj({
                'weights': aggregated_weights,
                'next_model': next_model
            })
            print(f"Sent weights to {self.address}, next model is {next_model} at {time.strftime('%H:%M:%S')}")


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
        
        print(f"Parameter server is running at {time.strftime('%H:%M:%S')}...")
        for thread in threads:
            thread.join()

def decide_model_for_next_training(performance_data):
    global current_round
    # Implement logic to decide the next model for each client based on performance_data
    # Example: return ['C', 'B', 'A'] based on some criteria

    # 1. Compute UCB scores for the current round
    client_scores = ucb_calculator.compute_scores(current_round)


    # switch k-v
    model_mapping_reverse = {v: k for k, v in model_mapping.items()}


    # 2. update training loss history 
    for c_perf in performance_data:
        c = c_perf['client_id'] - 1
        m = model_mapping_reverse[c_perf['model_name']]
        loss = c_perf['epoch_loss']
        ucb_calculator.update(c, m, loss, current_round)

    print(ucb_calculator.history)


    # 3. Call Ranklist-Multi-UCB to select clients and models
    selected_clients, assigned_models = ranklist_multi_ucb(
        client_scores=client_scores,
        all_clients=list(range(num_clients)),
        K=K,
        current_round=current_round,
        M=M
    )

    
    # Print results
    print(f"Selected Clients: {selected_clients}")

    assigned_models_mapping = [model_mapping[m] for m in assigned_models]
    color_print(f"Assigned Models:  {assigned_models_mapping}", 'yellow')

    print("Client Scores:")
    for c in selected_clients:
        scores = client_scores[c]
        print(f"  Client {c}: Model_A={scores[0]:.4f}, Model_B={scores[1]:.4f}, Model_C={scores[2]:.4f}")
    current_round += 1
    return assigned_models_mapping  

if __name__ == "__main__":
    ## UCB_Algo init
    # Parameter configuration
    num_clients = 3
    M = 3
    K = 3
    gamma = 0.9
    current_round = 0
    model_mapping = {0: "A", 1: "B", 2: "C"}

    # Generate fake data ratio (assuming different data distribution for each client)
    # np.random.seed(42)
    # data_ratio = {
    #     c: list(np.random.dirichlet(np.ones(M)))  # Dirichlet distribution ensures sum to 1
    #     for c in range(num_clients)
    # }
    # 手动设置固定比例（示例：每个模型在客户端上均匀分布）
    data_ratio = {
        0: [0.3333, 0.3333, 0.3334],  
        1: [0.3333, 0.3333, 0.3334],  
        2: [0.3333, 0.3333, 0.3334]  
    }

    # Initialize UCB calculator
    ucb_calculator = UCBCalculator(
        all_clients=list(range(num_clients)),
        num_models=M,
        data_ratio=data_ratio,
        gamma=gamma
    )
    ## UCB_Algo init done!

    server = ParameterServer()
    server.start()