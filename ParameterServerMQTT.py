import json
import torch
import random
from paho.mqtt import client as mqtt_client
from ollama import Client  # 导入ollama客户端
from model import ModelA, ModelB  # 确保模型的定义已正确导入

def color_print(text, color='black'):
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
        self.first_update = {'A': True, 'B': True}
        self.alpha = 0.1
        self.client_performance_data = {}  # Performance Dictionary
        try:
            self.ollama_client = Client(host='http://10.0.0.31:11434')  # Initialize the ollama client
            print("LLaMa client created successfully")
        except Exception as e:
            print(f"Error creating LLaMa client: {str(e)}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    def save_global_weights(self):
        torch.save(self.global_model_A.state_dict(), 'weights/global_model_A_weights.pth')
        torch.save(self.global_model_B.state_dict(), 'weights/global_model_B_weights.pth')
        print("Global weights saved.")

    def on_message(self, client, userdata, msg):
        try:
            message = json.loads(msg.payload.decode())
            model_name = message.get('model_name')
            client_id = message.get('client_id')

            if not model_name or not client_id:
                print(f"Received incomplete message: {message}")
                return

            model = self.global_model_A if model_name == 'A' else self.global_model_B
            client_params = {k: torch.tensor(v) for k, v in message['params'].items()}

            performance_data = message.get('performance')
            if performance_data:
                if client_id not in self.client_performance_data:
                    self.client_performance_data[client_id] = []
                self.client_performance_data[client_id].append(performance_data)
                print(f"Performance data from {client_id}: {performance_data}")

            # Update global model parameters
            with torch.no_grad():
                if self.first_update[model_name]:
                    model.load_state_dict(client_params)  # Load client parameters directly
                    self.first_update[model_name] = False
                else:
                    for param, new_param in zip(model.parameters(), client_params.values()):
                        param.data = param.data * (1.0 - self.alpha) + new_param.data.clone() * self.alpha

            self.save_global_weights()

            # Decide the model for the next training
            model_next_training = self.decide_model_for_next_training(client_id, model_name)
            print(f"Model for next training: {model_next_training}")

            # Send updated parameters to the client
            updated_params = {k: v.tolist() for k, v in model.state_dict().items()}
            self.publish_params(model_next_training, updated_params, client_id)

        except Exception as e:
            print(f"An error occurred processing the message: {e}")

    def publish_params(self, model_name, params, client_id):
        model_next_training = model_name

        # Select the appropriate model based on model_next_training
        if model_next_training == 'A':
            model = self.global_model_A
        else:
            model = self.global_model_B

        # Get the parameters of the selected model
        params = {k: v.tolist() for k, v in model.state_dict().items()}

        color_print(f'The server starts to send params to {client_id}.', color='red')
        message = json.dumps({"model_name": model_name, "params": params, "model_next_training": model_next_training, "client_id": client_id})
        self.client.publish(f"{topic_pull}_{client_id}", message)
        color_print('Sent.', color='green')

    def decide_model_for_next_training(self, client_id, model_name):
        model_descriptions = {
            'A': "Model A: Conv2d(1, 20, 7), ReLU, Conv2d(20, 40, 7), ReLU, MaxPool2d(2, 2), Flatten, Linear(40*5*5, 10).",
            'B': "Model B: Conv2d(1, 20, 7), ReLU, Conv2d(20, 40, 7), ReLU, Conv2d(40, 60, 5), ReLU, Conv2d(60, 80, 3), ReLU, MaxPool2d(2, 2), Flatten, Linear(80*5*5, 10)."
        }
        history_json = json.dumps(self.client_performance_data.get(client_id, []))
        prompt = (f"Client {client_id} is currently training {model_name}. Based on the following performance history: {history_json}, "
                f"and model structures {model_descriptions['A']} {model_descriptions['B']}, which model should be trained next? "
                f"Return only the letter A or B as the first character of your response.")
        messages = [{"role": "user", "content": prompt}]
        print(f"History for client {client_id}: {history_json}")  # Print history data
        response = self.ollama_client.chat(model='llama3', messages=messages)
        print(f"Response from llama3: {response}")  # Print response from llama3
        decision = None
        decision_source = None  # Used to track the source of the decision
        if 'message' in response and 'content' in response['message']:
            first_char = response['message']['content'].strip()[0].upper()
            if first_char in ['A', 'B']:
                decision = first_char
                decision_source = 'llama3'  # If the decision comes from llama3, set the decision source as 'llama3'
            else:
                print(f"Invalid decision from llama3: {first_char}")
        if decision is None:
            print("No valid decision received from llama, falling back to random choice")
            decision = ['A', 'B'][random.randint(0, 1)]
            decision_source = 'random'  # If the decision comes from random choice, set the decision source as 'random'
        print(f"Final decision for client {client_id}: {decision} (source: {decision_source})")  # Print the final decision and decision source
        return decision

if __name__ == "__main__":
    server = ParameterServerMQTT()
