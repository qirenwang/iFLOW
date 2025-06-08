import zmq
import torch
import threading
import random
from model import ModelSSDlite as ModelA
from model import ModelFasterRCNNMobileNet as ModelB
from model import ModelFasterRCNNResNet as ModelC

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

    def aggregate_weights(self, model_name, client_weights):
        """聚合指定模型的权重"""
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
        """获取模型权重"""
        with self.lock:
            return self.model_weights[model_name]

class ClientHandler(threading.Thread):
    def __init__(self, client_id, address, model_manager):
        super().__init__()
        self.client_id = client_id
        self.address = address
        self.model_manager = model_manager
        self.context = zmq.Context()
        
        # 设置接收socket
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind(f"tcp://*:{5555 + client_id}")
        
        # 设置发送socket
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.connect(f"tcp://{address}:{6666 + client_id}")

    def run(self):
        while True:

            # 接收客户端的权重
            msg = self.receiver.recv_pyobj()
            current_model = msg['model_name']
            weights = msg['weights']
            performance = msg['performance']
            print(f"Received weights from {self.address}, model is {current_model}")
            print(performance)
            # 聚合权重
            self.model_manager.aggregate_weights(current_model, weights)
            
            # 随机选择下一个模型
            next_model = random.choice(['A', 'B', 'C'])
            
            # 发送next_model后的权重
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
            1: "10.0.0.2",
            # 1: "10.0.0.15",
            # 2: "10.0.0.14",
            # 3: "10.0.0.4"
        }

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




# class ModelManager:
#     def __init__(self):
#         self.models = {
#             'A': ModelA(),
#             'B': ModelB(),
#             'C': ModelC()
#         }
#         self.model_weights = {
#             'A': self.models['A'].state_dict(),
#             'B': self.models['B'].state_dict(),
#             'C': self.models['C'].state_dict()
#         }
#         self.lock = threading.Lock()
        
#         # 添加全局轮次跟踪
#         self.global_rounds = {
#             'A': 0,
#             'B': 0,
#             'C': 0
#         }
        
#         # 创建目录和清理日志文件
#         os.makedirs("server_logs", exist_ok=True)
        
#         # 清理训练历史日志
#         with open("server_logs/training_history.txt", "w") as f:
#             f.write("")  # 清空文件内容
            
#         # 创建聚合日志文件
#         with open("server_logs/aggregation_log.txt", "w") as f:
#             f.write("# Staleness-aware Aggregation Log\n")
#             f.write("Model,Client,Global_Round,Client_Round,Staleness,Weight,Timestamp\n")

#     def get_current_global_round(self, model_name):
#         return self.global_rounds[model_name]

#     def update_global_round(self, model_name):
#         self.global_rounds[model_name] += 1

#     def aggregate_weights(self, model_name, client_weights, client_performance):
#         with self.lock:
#             current_weights = self.model_weights[model_name]
#             # 获取全局轮次和客户端轮次（保留跟踪，但仅用于日志）
#             current_global_round = self.get_current_global_round(model_name)
#             client_round = client_performance['epoch']
#             staleness = max(0, current_global_round - client_round)
#             # 简单的FedAvg策略 - 固定0.5权重
#             alpha = 0.5
#             # 记录到聚合日志
#             timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#             with open("server_logs/aggregation_log.txt", "a") as f:
#                 f.write(f"{model_name},{client_performance['client_id']},{current_global_round},{client_round},{staleness},{alpha},{timestamp}\n")
#             # 使用聚合机制
#             aggregated_weights = {}
#             for key in current_weights.keys():
#                 client_weight = client_weights[key].float()
#                 current_weight = current_weights[key].float()
#                 aggregated_weights[key] = (1 - alpha) * current_weight + alpha * client_weight
#             # 更新模型权重
#             self.model_weights[model_name] = aggregated_weights
#             # 更新全局轮次
#             self.update_global_round(model_name)

#     def get_weights(self, model_name):
#         with self.lock:
#             return self.model_weights[model_name]
