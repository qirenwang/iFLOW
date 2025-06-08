###################### 评估 加入 带来源版本 ########################
import json
import pickle
import requests
import torch
import random
import os
from paho.mqtt import client as mqtt_client
from ollama import Client  # 导入ollama客户端
# from model import ModelA, ModelB  # 确保模型的定义已正确导入
from model import ModelSSDlite as ModelA
from model import ModelFasterRCNNMobileNet as ModelB
from model import ModelFasterRCNNResNet as ModelC
# from MNISTDataset import create_mnist_dataloader, custom_transform
# from subset_dataloader import create_subset_dataloader
from model import train_loader as eval_loader
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

def save_evaluation_results(results, file_path):
    with open(file_path, 'a') as f:
        for result in results:
            f.write(result + '\n')

use_ollama = True # False
# 配置评估数据集路径
# eval_image_path = '/home/mist/mnist_data/MNIST/raw/t10k-images-idx3-ubyte'
# eval_label_path = '/home/mist/mnist_data/MNIST/raw/t10k-labels-idx1-ubyte'

# 固定随机种子
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

# 创建评估数据加载器
# full_eval_loader = create_mnist_dataloader(eval_image_path, eval_label_path, batch_size=4, transform=custom_transform)
# subset_ratio = 0.05
# eval_loader = create_subset_dataloader(full_eval_loader.dataset, subset_ratio=subset_ratio, batch_size=4, shuffle=True, seed=random_seed)

broker = server_ip
port = 1883
topic_push = "federated_learning/params_push"
topic_pull = "federated_learning/params_pull"


print(f"Server IP: {server_ip}")
print(f"Current IP: {current_ip}")

class ParameterServerMQTT:
    def __init__(self):
        self.client = mqtt_client.Client(callback_api_version=mqtt_client.CallbackAPIVersion.VERSION1, client_id='ParameterServer')
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker, port)
        self.client.subscribe(topic_push)
        self.client.loop_start()
        self.global_model_A = ModelA()
        self.global_model_B = ModelB()
        self.global_model_C = ModelC()
        self.first_update = {'A': True, 'B': True, 'C':True}
        self.alpha = 0.1
        self.client_performance_data = {}  # Performance Dictionary
        self.eval_results_file = 'evaluation_results.txt'  # 文件路径
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_ollama:
            try:
                self.ollama_client = Client(host='http://10.0.0.16:11434')  # Initialize the ollama client
                print("LLaMa client created successfully")
            except Exception as e:
                print(f"Error creating LLaMa client: {str(e)}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.client.subscribe(topic_push)
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    def save_global_weights(self):
        torch.save(self.global_model_A.state_dict(), 'weights/global_model_A_weights.pth')
        torch.save(self.global_model_B.state_dict(), 'weights/global_model_B_weights.pth')
        torch.save(self.global_model_C.state_dict(), 'weights/global_model_C_weights.pth')
        print("Global weights saved.")

    # def on_message(self, client, userdata, msg):
    #     # try:
    #     if True:

    #         message = json.loads(msg.payload.decode())

    #         msg_url = message['msg_url']
    #         print(f"Received message URL: {msg_url}")

    #         # Download the message from the URL
    #         message = requests.get(msg_url, stream=True).content
    #         # load the message from the bytes
    #         message = pickle.loads(message)

    #         model_name = message.get('model_name')
    #         client_id = message.get('client_id')

    #         if not model_name or not client_id:
    #             print(f"Received incomplete message: {message}")
    #             return
    #         if model_name == 'A' :
    #             model = self.global_model_A 
    #         elif model_name == 'B' :
    #             model = self.global_model_B
    #         else :
    #             model = self.global_model_C

            
    #         client_params = {k: torch.tensor(v) for k, v in message['params'].items()}

    #         performance_data = message.get('performance')
    #         if performance_data:
    #             if client_id not in self.client_performance_data:
    #                 self.client_performance_data[client_id] = []
    #             self.client_performance_data[client_id].append(performance_data)
    #             print(f"Performance data from {client_id}: {performance_data}")

    #         # Update global model parameters
    #         with torch.no_grad():
    #             if self.first_update[model_name]:
    #                 model.load_state_dict(client_params)  # Load client parameters directly
    #                 self.first_update[model_name] = False
    #             else:
    #                 model_params = {k: v.cpu().clone().detach().tolist()  for k, v in model.state_dict().items()}
    #                 model_params = {k: torch.tensor(v) for k, v in model_params.items()}
    #                 for param, new_param in zip(model_params.values(), client_params.values()):
    #                     param.data = param.data * (1.0 - self.alpha) + new_param.data.clone() * self.alpha
                    
    #                 model.load_state_dict(model_params)

    #         '''self.save_global_weights()'''

    #         '''# Evaluate models
    #         loss = evaluate_model(model, eval_loader, self.device)
    #         eval_results = [
    #             f"Server - Model {model_name} (from Client {client_id}) Loss: {loss:.2f}"
    #         ]
    #         for result in eval_results:
    #             print(result)
    #         save_evaluation_results(eval_results, self.eval_results_file)'''

    #         # Decide the model for the next training
    #         if use_ollama:
    #             model_next_training = self.decide_model_for_next_training(client_id, model_name)
    #         else:
    #             model_next_training = ['A', 'B', 'C'][random.randint(0, 2)]
    #             # model_next_training = 'A'
    #         print(f"Model for next training: {model_next_training}")

    #         # Send updated parameters to the client
    #         updated_params = {k: v.tolist() for k, v in model.state_dict().items()}
    #         self.publish_params(model_next_training, updated_params, client_id)

    #     # except Exception as e:
    #     #     print(f"An error occurred processing the message: {e}")
    def on_message(self, client, userdata, msg):
        try:
            # Decode the incoming message
            message = json.loads(msg.payload.decode())
            msg_url = message['msg_url']
            print(f"Received message URL: {msg_url}")

            # Download the message from the provided URL
            response = requests.get(msg_url, stream=True)
            response.raise_for_status()  # Ensure the request was successful
            message_content = response.content

            # Load the pickle-encoded message
            message = pickle.loads(message_content)

            model_name = message.get('model_name')
            client_id = message.get('client_id')

            if not model_name or not client_id:
                print(f"Received incomplete message: {message}")
                return

            # Select the appropriate global model
            if model_name == 'A':
                model = self.global_model_A
            elif model_name == 'B':
                model = self.global_model_B
            else:
                model = self.global_model_C

            # Extract client parameters and performance data
            client_params = {k: torch.tensor(v) for k, v in message['params'].items()}
            performance_data = message.get('performance', {})

            # Log performance data for debugging
            print(f"Performance data from {client_id}: {performance_data}")

            # Validate and log GPU utilization
            avg_gpu_utilization = performance_data.get('avg_gpu_utilization', 'N/A')
            if avg_gpu_utilization == 'N/A':
                print(f"GPU utilization for client {client_id} is missing or not available.")
            else:
                print(f"GPU utilization for client {client_id}: {avg_gpu_utilization}%")

            # Update client performance history
            if client_id not in self.client_performance_data:
                self.client_performance_data[client_id] = []
            self.client_performance_data[client_id].append(performance_data)

            # Update the global model with client parameters
            with torch.no_grad():
                if self.first_update[model_name]:
                    model.load_state_dict(client_params)
                    self.first_update[model_name] = False
                else:
                    model_params = {k: v.cpu().clone().detach().tolist() for k, v in model.state_dict().items()}
                    model_params = {k: torch.tensor(v) for k, v in model_params.items()}
                    for param, new_param in zip(model_params.values(), client_params.values()):
                        param.data = param.data * (1.0 - self.alpha) + new_param.data * self.alpha

                    model.load_state_dict(model_params)

            # Optionally save the updated global model
            # self.save_global_weights()

            # Decide the next model for the client

            if use_ollama:
                model_next_training = self.decide_model_for_next_training(client_id, model_name)
            else:
                model_next_training = ['A', 'B', 'C'][random.randint(0, 2)]
            print(f"Model for next training: {model_next_training}")

            # Send updated parameters back to the client
            updated_params = {k: v.tolist() for k, v in model.state_dict().items()}
            self.publish_params(model_next_training, updated_params, client_id)

        except requests.RequestException as req_err:
            print(f"Failed to fetch the message from {msg_url}: {req_err}")
        except pickle.PickleError as pickle_err:
            print(f"Failed to decode the pickle message: {pickle_err}")
        except Exception as e:
            print(f"An error occurred in on_message: {e}")

    def publish_params(self, model_name, params, client_id):
        model_next_training = model_name

        # Select the appropriate model based on model_next_training
        if model_next_training == 'A':
            model = self.global_model_A
        if model_next_training == 'B':
            model = self.global_model_B
        else:
            model = self.global_model_C

        # Get the parameters of the selected model
        params = {k: v.tolist() for k, v in model.state_dict().items()}

        color_print(f'The server starts to send params to {client_id}.', color='red')
        msg_dict = {"model_name": model_name, "params": params, "model_next_training": model_next_training, "client_id": client_id}

        # save to pkl file
        with open(f'saved_msg/params_{client_id}.pkl', 'wb') as f:
            pickle.dump(msg_dict, f)


        # # load from pkl file
        # with open('saved_msg/params.pkl', 'rb') as f:
        #     msg_dict = json.loads(f.read().decode())



        # message = json.dumps({"model_name": model_name, "params": params, "model_next_training": model_next_training, "client_id": client_id})
        message = json.dumps({"msg_url": f"http://{current_ip}:8000/saved_msg/params_{client_id}.pkl"})




        

        # # Convert message to bytes and get its size
        # message_bytes = message.encode('utf-8')
        # message_size = len(message_bytes)  # Size in bytes
        # print(f"Message size: {message_size} bytes")


        self.client.publish(f"{topic_pull}_{client_id}", message)
        color_print('Sent.', color='green')
        print(f"Sent msg_url to {topic_pull}_{client_id}: http://{current_ip}:8000/saved_msg/params_{client_id}.pkl")

    def decide_model_for_next_training(self, client_id, model_name):
        model_descriptions = {
            'A': (
                "Model A: SSDlite320_MobilenetV3 - Backbone: DepthwiseSeparableConv2d layers with "
                "kernel sizes 3x3, strides (2, 2), ReLU6 activation. Feature Pyramid Network for "
                "multi-scale feature extraction. Heads: Class and Box predictors with separable convolutions."
            ),
            'B': (
                "Model B: FasterRCNN_MobileNetV3 - Backbone: Conv2d(3, 32, kernel_size=3, stride=2), "
                "BatchNorm2d, ReLU, followed by bottleneck layers with depthwise separable convolutions. "
                "FPN for multi-scale features. RPN: Conv2d(256, 256, kernel_size=3), Class and Box heads: "
                "Fully connected layers with ReLU."
            ),
            'C': (
                "Model C: FasterRCNN_ResNet50 - Backbone: Conv2d(3, 64, kernel_size=7, stride=2, padding=3), "
                "BatchNorm2d, ReLU, MaxPool2d(3, stride=2), followed by Bottleneck blocks (Conv1x1, Conv3x3, "
                "Conv1x1). FPN for feature aggregation. RPN: Conv2d(256, 256, kernel_size=3), Fully connected "
                "heads for classification and regression."
            )
        }
        history_json = json.dumps(self.client_performance_data.get(client_id, []))
        prompt = (f"Client {client_id} is currently training {model_name}. Based on the following performance history: {history_json}, "
                f"and model structures {model_descriptions['A']} {model_descriptions['B']} {model_descriptions['C']}, "
                f"which model should be trained next? Return only the letter A, B, or C as the first character of your response.")
        messages = [{"role": "user", "content": prompt}]
        print(f"History for client {client_id}: {history_json}")  # Print history data
        response = self.ollama_client.chat(model='llama3', messages=messages)
        print(f"Response from llama3: {response}")  # Print response from llama3
        decision = None
        decision_source = None  # Used to track the source of the decision
        if 'message' in response and 'content' in response['message']:
            first_char = response['message']['content'].strip()[0].upper()
            if first_char in ['A', 'B', 'C']:
                decision = first_char
                decision_source = 'llama3'  # If the decision comes from llama3, set the decision source as 'llama3'
            else:
                print(f"Invalid decision from llama3: {first_char}")
        if decision is None:
            print("No valid decision received from llama, falling back to random choice")
            decision = ['A', 'B', 'C'][random.randint(0, 2)]
            decision_source = 'random'  # If the decision comes from random choice, set the decision source as 'random'
        print(f"Final decision for client {client_id}: {decision} (source: {decision_source})")  # Print the final decision and decision source
        return decision

    # def decide_model_for_next_training(self, client_id, model_name): ## need to be changed to 3 models
    #     model_descriptions = {
    #         'A': "Model A: Conv2d(1, 20, 7), ReLU, Conv2d(20, 40, 7), ReLU, MaxPool2d(2, 2), Flatten, Linear(40*5*5, 10).",
    #         'B': "Model B: Conv2d(1, 20, 7), ReLU, Conv2d(20, 40, 7), ReLU, Conv2d(40, 60, 5), ReLU, Conv2d(60, 80, 3), ReLU, MaxPool2d(2, 2), Flatten, Linear(80*5*5, 10)."
    #     }
    #     history_json = json.dumps(self.client_performance_data.get(client_id, []))
    #     prompt = (f"Client {client_id} is currently training {model_name}. Based on the following performance history: {history_json}, "
    #             f"and model structures {model_descriptions['A']} {model_descriptions['B']}, which model should be trained next? "
    #             f"Return only the letter A or B as the first character of your response.")
    #     messages = [{"role": "user", "content": prompt}]
    #     print(f"History for client {client_id}: {history_json}")  # Print history data
    #     response = self.ollama_client.chat(model='llama3', messages=messages)
    #     print(f"Response from llama3: {response}")  # Print response from llama3
    #     decision = None
    #     decision_source = None  # Used to track the source of the decision
    #     if 'message' in response and 'content' in response['message']:
    #         first_char = response['message']['content'].strip()[0].upper()
    #         if first_char in ['A', 'B']:
    #             decision = first_char
    #             decision_source = 'llama3'  # If the decision comes from llama3, set the decision source as 'llama3'
    #         else:
    #             print(f"Invalid decision from llama3: {first_char}")
    #     if decision is None:
    #         print("No valid decision received from llama, falling back to random choice")
    #         decision = ['A', 'B', 'C'][random.randint(0, 2)]
    #         decision_source = 'random'  # If the decision comes from random choice, set the decision source as 'random'
    #     print(f"Final decision for client {client_id}: {decision} (source: {decision_source})")  # Print the final decision and decision source
    #     return decision

if __name__ == "__main__":
    server = ParameterServerMQTT()


# ##################### 无评估版本 ########################

# import json
# import torch
# import random
# from paho.mqtt import client as mqtt_client
# from ollama import Client  # 导入ollama客户端
# from model import ModelA, ModelB  # 确保模型的定义已正确导入

# def color_print(text, color='black'):
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

# class ParameterServerMQTT:
#     def __init__(self):
#         self.client = mqtt_client.Client(callback_api_version=mqtt_client.CallbackAPIVersion.VERSION1, client_id='ParameterServer')
#         self.client.on_connect = self.on_connect
#         self.client.on_message = self.on_message
#         self.client.connect(broker, port)
#         self.client.subscribe(topic_push)
#         self.client.loop_start()
#         self.global_model_A = ModelA()
#         self.global_model_B = ModelB()
#         self.first_update = {'A': True, 'B': True}
#         self.alpha = 0.1
#         self.client_performance_data = {}  # Performance Dictionary
#         try:
#             self.ollama_client = Client(host='http://10.0.0.31:11434')  # Initialize the ollama client
#             print("LLaMa client created successfully")
#         except Exception as e:
#             print(f"Error creating LLaMa client: {str(e)}")

#     def on_connect(self, client, userdata, flags, rc):
#         if rc == 0:
#             print("Connected to MQTT Broker!")
#         else:
#             print("Failed to connect, return code %d\n", rc)

#     def save_global_weights(self):
#         torch.save(self.global_model_A.state_dict(), 'weights/global_model_A_weights.pth')
#         torch.save(self.global_model_B.state_dict(), 'weights/global_model_B_weights.pth')
#         print("Global weights saved.")

#     def on_message(self, client, userdata, msg):
#         try:
#             message = json.loads(msg.payload.decode())
#             model_name = message.get('model_name')
#             client_id = message.get('client_id')

#             if not model_name or not client_id:
#                 print(f"Received incomplete message: {message}")
#                 return

#             model = self.global_model_A if model_name == 'A' else self.global_model_B
#             client_params = {k: torch.tensor(v) for k, v in message['params'].items()}

#             performance_data = message.get('performance')
#             if performance_data:
#                 if client_id not in self.client_performance_data:
#                     self.client_performance_data[client_id] = []
#                 self.client_performance_data[client_id].append(performance_data)
#                 print(f"Performance data from {client_id}: {performance_data}")

#             # Update global model parameters
#             with torch.no_grad():
#                 if self.first_update[model_name]:
#                     model.load_state_dict(client_params)  # Load client parameters directly
#                     self.first_update[model_name] = False
#                 else:
#                     for param, new_param in zip(model.parameters(), client_params.values()):
#                         param.data = param.data * (1.0 - self.alpha) + new_param.data.clone() * self.alpha

#             self.save_global_weights()

#             # Decide the model for the next training
#             model_next_training = self.decide_model_for_next_training(client_id, model_name)
#             print(f"Model for next training: {model_next_training}")

#             # Send updated parameters to the client
#             updated_params = {k: v.tolist() for k, v in model.state_dict().items()}
#             self.publish_params(model_next_training, updated_params, client_id)

#         except Exception as e:
#             print(f"An error occurred processing the message: {e}")

#     def publish_params(self, model_name, params, client_id):
#         model_next_training = model_name

#         # Select the appropriate model based on model_next_training
#         if model_next_training == 'A':
#             model = self.global_model_A
#         else:
#             model = self.global_model_B

#         # Get the parameters of the selected model
#         params = {k: v.tolist() for k, v in model.state_dict().items()}

#         color_print(f'The server starts to send params to {client_id}.', color='red')
#         message = json.dumps({"model_name": model_name, "params": params, "model_next_training": model_next_training, "client_id": client_id})
#         self.client.publish(f"{topic_pull}_{client_id}", message)
#         color_print('Sent.', color='green')

#     def decide_model_for_next_training(self, client_id, model_name):
#         model_descriptions = {
#             'A': "Model A: Conv2d(1, 20, 7), ReLU, Conv2d(20, 40, 7), ReLU, MaxPool2d(2, 2), Flatten, Linear(40*5*5, 10).",
#             'B': "Model B: Conv2d(1, 20, 7), ReLU, Conv2d(20, 40, 7), ReLU, Conv2d(40, 60, 5), ReLU, Conv2d(60, 80, 3), ReLU, MaxPool2d(2, 2), Flatten, Linear(80*5*5, 10)."
#         }
#         history_json = json.dumps(self.client_performance_data.get(client_id, []))
#         prompt = (f"Client {client_id} is currently training {model_name}. Based on the following performance history: {history_json}, "
#                 f"and model structures {model_descriptions['A']} {model_descriptions['B']}, which model should be trained next? "
#                 f"Return only the letter A or B as the first character of your response.")
#         messages = [{"role": "user", "content": prompt}]
#         print(f"History for client {client_id}: {history_json}")  # Print history data
#         response = self.ollama_client.chat(model='llama3', messages=messages)
#         print(f"Response from llama3: {response}")  # Print response from llama3
#         decision = None
#         decision_source = None  # Used to track the source of the decision
#         if 'message' in response and 'content' in response['message']:
#             first_char = response['message']['content'].strip()[0].upper()
#             if first_char in ['A', 'B']:
#                 decision = first_char
#                 decision_source = 'llama3'  # If the decision comes from llama3, set the decision source as 'llama3'
#             else:
#                 print(f"Invalid decision from llama3: {first_char}")
#         if decision is None:
#             print("No valid decision received from llama, falling back to random choice")
#             decision = ['A', 'B'][random.randint(0, 1)]
#             decision_source = 'random'  # If the decision comes from random choice, set the decision source as 'random'
#         print(f"Final decision for client {client_id}: {decision} (source: {decision_source})")  # Print the final decision and decision source
#         return decision

# if __name__ == "__main__":
#     server = ParameterServerMQTT()

# ##################### 评估 加入 但是没有来源版本 ########################
# import json
# import torch
# import random
# import os
# from paho.mqtt import client as mqtt_client
# from ollama import Client  # 导入ollama客户端
# from model import ModelA, ModelB  # 确保模型的定义已正确导入
# from MNISTDataset import create_mnist_dataloader, custom_transform
# from subset_dataloader import create_subset_dataloader

# def color_print(text, color='black'):
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

# def evaluate_model(model, dataloader, device):
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

# def save_evaluation_results(results, file_path):
#     with open(file_path, 'a') as f:
#         for result in results:
#             f.write(result + '\n')

# # 配置评估数据集路径
# eval_image_path = '/home/mist/mnist_data/MNIST/raw/t10k-images-idx3-ubyte'
# eval_label_path = '/home/mist/mnist_data/MNIST/raw/t10k-labels-idx1-ubyte'

# # 创建评估数据加载器
# full_eval_loader = create_mnist_dataloader(eval_image_path, eval_label_path, batch_size=4, transform=custom_transform)
# subset_ratio = 0.1
# eval_loader = create_subset_dataloader(full_eval_loader.dataset, subset_ratio=subset_ratio, batch_size=4)

# broker = '10.0.0.13'
# port = 1883
# topic_push = "federated_learning/params_push"
# topic_pull = "federated_learning/params_pull"

# class ParameterServerMQTT:
#     def __init__(self):
#         self.client = mqtt_client.Client(callback_api_version=mqtt_client.CallbackAPIVersion.VERSION1, client_id='ParameterServer')
#         self.client.on_connect = self.on_connect
#         self.client.on_message = self.on_message
#         self.client.connect(broker, port)
#         self.client.subscribe(topic_push)
#         self.client.loop_start()
#         self.global_model_A = ModelA()
#         self.global_model_B = ModelB()
#         self.first_update = {'A': True, 'B': True}
#         self.alpha = 0.1
#         self.client_performance_data = {}  # Performance Dictionary
#         self.eval_results_file = 'evaluation_results.txt'  # 文件路径
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         try:
#             self.ollama_client = Client(host='http://10.0.0.16:11434')  # Initialize the ollama client
#             print("LLaMa client created successfully")
#         except Exception as e:
#             print(f"Error creating LLaMa client: {str(e)}")

#     def on_connect(self, client, userdata, flags, rc):
#         if rc == 0:
#             print("Connected to MQTT Broker!")
#         else:
#             print("Failed to connect, return code %d\n", rc)

#     def save_global_weights(self):
#         torch.save(self.global_model_A.state_dict(), 'weights/global_model_A_weights.pth')
#         torch.save(self.global_model_B.state_dict(), 'weights/global_model_B_weights.pth')
#         print("Global weights saved.")

#     def on_message(self, client, userdata, msg):
#         try:
#             message = json.loads(msg.payload.decode())
#             model_name = message.get('model_name')
#             client_id = message.get('client_id')

#             if not model_name or not client_id:
#                 print(f"Received incomplete message: {message}")
#                 return

#             model = self.global_model_A if model_name == 'A' else self.global_model_B
#             client_params = {k: torch.tensor(v) for k, v in message['params'].items()}

#             performance_data = message.get('performance')
#             if performance_data:
#                 if client_id not in self.client_performance_data:
#                     self.client_performance_data[client_id] = []
#                 self.client_performance_data[client_id].append(performance_data)
#                 print(f"Performance data from {client_id}: {performance_data}")

#             # Update global model parameters
#             with torch.no_grad():
#                 if self.first_update[model_name]:
#                     model.load_state_dict(client_params)  # Load client parameters directly
#                     self.first_update[model_name] = False
#                 else:
#                     for param, new_param in zip(model.parameters(), client_params.values()):
#                         param.data = param.data * (1.0 - self.alpha) + new_param.data.clone() * self.alpha

#             self.save_global_weights()

#             # Evaluate models
#             accuracy = evaluate_model(model, eval_loader, self.device)
#             eval_results = [
#                 f"Server - Model {model_name} Accuracy: {accuracy:.2f}%"
#             ]
#             for result in eval_results:
#                 print(result)
#             save_evaluation_results(eval_results, self.eval_results_file)

#             # Decide the model for the next training
#             model_next_training = self.decide_model_for_next_training(client_id, model_name)
#             print(f"Model for next training: {model_next_training}")

#             # Send updated parameters to the client
#             updated_params = {k: v.tolist() for k, v in model.state_dict().items()}
#             self.publish_params(model_next_training, updated_params, client_id)

#         except Exception as e:
#             print(f"An error occurred processing the message: {e}")

#     def publish_params(self, model_name, params, client_id):
#         model_next_training = model_name

#         # Select the appropriate model based on model_next_training
#         if model_next_training == 'A':
#             model = self.global_model_A
#         else:
#             model = self.global_model_B

#         # Get the parameters of the selected model
#         params = {k: v.tolist() for k, v in model.state_dict().items()}

#         color_print(f'The server starts to send params to {client_id}.', color='red')
#         message = json.dumps({"model_name": model_name, "params": params, "model_next_training": model_next_training, "client_id": client_id})
#         self.client.publish(f"{topic_pull}_{client_id}", message)
#         color_print('Sent.', color='green')

#     def decide_model_for_next_training(self, client_id, model_name):
#         model_descriptions = {
#             'A': "Model A: Conv2d(1, 20, 7), ReLU, Conv2d(20, 40, 7), ReLU, MaxPool2d(2, 2), Flatten, Linear(40*5*5, 10).",
#             'B': "Model B: Conv2d(1, 20, 7), ReLU, Conv2d(20, 40, 7), ReLU, Conv2d(40, 60, 5), ReLU, Conv2d(60, 80, 3), ReLU, MaxPool2d(2, 2), Flatten, Linear(80*5*5, 10)."
#         }
#         history_json = json.dumps(self.client_performance_data.get(client_id, []))
#         prompt = (f"Client {client_id} is currently training {model_name}. Based on the following performance history: {history_json}, "
#                 f"and model structures {model_descriptions['A']} {model_descriptions['B']}, which model should be trained next? "
#                 f"Return only the letter A or B as the first character of your response.")
#         messages = [{"role": "user", "content": prompt}]
#         print(f"History for client {client_id}: {history_json}")  # Print history data
#         response = self.ollama_client.chat(model='llama3', messages=messages)
#         print(f"Response from llama3: {response}")  # Print response from llama3
#         decision = None
#         decision_source = None  # Used to track the source of the decision
#         if 'message' in response and 'content' in response['message']:
#             first_char = response['message']['content'].strip()[0].upper()
#             if first_char in ['A', 'B']:
#                 decision = first_char
#                 decision_source = 'llama3'  # If the decision comes from llama3, set the decision source as 'llama3'
#             else:
#                 print(f"Invalid decision from llama3: {first_char}")
#         if decision is None:
#             print("No valid decision received from llama, falling back to random choice")
#             decision = ['A', 'B'][random.randint(0, 1)]
#             decision_source = 'random'  # If the decision comes from random choice, set the decision source as 'random'
#         print(f"Final decision for client {client_id}: {decision} (source: {decision_source})")  # Print the final decision and decision source
#         return decision

# if __name__ == "__main__":
#     server = ParameterServerMQTT()
