import json
import re
import random
from flask import Flask, request, jsonify
from ollama import Client

app = Flask(__name__)

# Ollama client
ollama_client = Client(host='http://10.0.0.44:11434')

# Store client performance data
client_performance_data = {}

def decide_model_for_next_training(client_id, model_name):
    # Model descriptions
    model_descriptions = {
        'A': "Model A: LSTM Model for Driving Intent Prediction",
        'B': "Model B: FasterRCNN_MobileNetV3 for Object Detection",
        'C': "Model C: LightWeight UNet for Lane Detection"
    }

    # Calculate training counts for each model
    model_counts = {'A': 0, 'B': 0, 'C': 0}
    for record in client_performance_data.get(client_id, []):
        model = record['model_name']
        if model in model_counts:
            model_counts[model] += 1

    # Prepare history JSON
    history_json = json.dumps(client_performance_data.get(client_id, []))

    # Build the prompt
    prompt = (
        f"Based on the following performance history: {history_json} for client {client_id} currently training {model_name}, "
        f"return exactly one letter (A, B, or C) to indicate which model to train next. The letter should be chosen based on the following strict priority order:\n"
        f"1. **Choose the model with the lowest training count** to maintain a balanced distribution. Here are the current training counts:\n"
        f"   - Model A: {model_counts['A']} times\n"
        f"   - Model B: {model_counts['B']} times\n"
        f"   - Model C: {model_counts['C']} times\n"
        f"2. **Avoid training the same model consecutively**. The last trained model was {model_name}.\n"
        f"3. Consider the model structures: {model_descriptions['A']}, {model_descriptions['B']}, {model_descriptions['C']}.\n\n"
        f"Your goal is to ensure that models A, B, and C are trained as evenly as possible. "
        f"Return only the letter (A, B, or C) with no additional text, explanation, or punctuation."
    )

    # Print the prompt for debugging
    print("Generated prompt:")
    print(prompt)

    # Call Ollama
    try:
        response = ollama_client.chat(model='llama3.3', messages=[{"role": "user", "content": prompt}])
        print(f"Response from Ollama: {response}")

        if 'message' in response and 'content' in response['message']:
            # Extract the first valid letter (A, B, or C) using regex
            match = re.search(r'[A-C]', response['message']['content'])
            if match:
                selected_model = match.group(0)
                print("Decision source: Ollama")
                return selected_model
            else:
                print(f"Invalid decision from Ollama: {response['message']['content']}")
    except Exception as e:
        print(f"Error calling Ollama: {e}")

    # Fallback to random choice if Ollama fails
    print("Decision source: Random")
    print("Ollama failed, falling back to random choice.")
    return random.choice(['A', 'B', 'C'])

@app.route('/next_model_decision', methods=['POST'])
def next_model_decision():
    data = request.get_json()

    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing "prompt" in request data'}), 400

    prompt = data['prompt']
    print(f"Received prompt: {prompt}")

    client_id = prompt['client_id']
    model_name = prompt['model_name']

    # Update client performance history
    client_performance_data.setdefault(client_id, []).append(prompt)

    # Decide the next model
    next_model = decide_model_for_next_training(client_id, model_name)
    print(f"Model for next training: {next_model}")

    return jsonify({'next_model': next_model})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# import json
# from flask import Flask, request, jsonify
# import random
# from ollama import Client

# app = Flask(__name__)

# # Flag to decide whether to use Ollama or fallback to random if Ollama fails
# # use_ollama = True  # Set this to False to always use random

# ollama_client = Client(host='http://10.0.0.44:11434')
# client_performance_data = {}

# def decide_model_for_next_training(client_id, model_name):
#     # model_descriptions = {
#     #     'A': "Model A: SSDlite320_MobilenetV3",
#     #     'B': "Model B: FasterRCNN_MobileNetV3",
#     #     'C': "Model C: FasterRCNN_ResNet50"
#     # }
#     model_descriptions = {
#     'A': "Model A: LSTM Model for Driving Intent Prediction",
#     'B': "Model B: FasterRCNN_MobileNetV3 for Object Detection",
#     'C': "Model C: LightWeight UNet for Lane Detection"
# }
#     # 计算每个模型的训练次数
#     model_counts = {'A': 0, 'B': 0, 'C': 0}
#     for record in client_performance_data.get(client_id, []):
#         model = record['model_name']
#         if model in model_counts:
#             model_counts[model] += 1

#     # 创建训练次数的摘要
#     counts_summary = (
#         f"Training counts:\n"
#         f"- Model A: {model_counts['A']} times\n"
#         f"- Model B: {model_counts['B']} times\n"
#         f"- Model C: {model_counts['C']} times\n"
#     )
#     # model_descriptions = {
#     # 'A': (
#     # "Model A: SSDlite320_MobilenetV3 - Backbone: DepthwiseSeparableConv2d layers with "
#     # "kernel sizes 3x3, strides (2, 2), ReLU6 activation. Feature Pyramid Network for "
#     # "multi-scale feature extraction. Heads: Class and Box predictors with separable convolutions."
#     # ),
#     # 'B': (
#     # "Model B: FasterRCNN_MobileNetV3 - Backbone: Conv2d(3, 32, kernel_size=3, stride=2), "
#     # "BatchNorm2d, ReLU, followed by bottleneck layers with depthwise separable convolutions. "
#     # "FPN for multi-scale features. RPN: Conv2d(256, 256, kernel_size=3), Class and Box heads: "
#     # "Fully connected layers with ReLU."
#     # ),
#     # 'C': (
#     # "Model C: FasterRCNN_ResNet50 - Backbone: Conv2d(3, 64, kernel_size=7, stride=2, padding=3), "
#     # "BatchNorm2d, ReLU, MaxPool2d(3, stride=2), followed by Bottleneck blocks (Conv1x1, Conv3x3, "
#     # "Conv1x1). FPN for feature aggregation. RPN: Conv2d(256, 256, kernel_size=3), Fully connected "
#     # "heads for classification and regression."
#     # )
#     # }

#     history_json = json.dumps(client_performance_data.get(client_id, []))

#     # prompt = (f"Client {client_id} is currently training {model_name}. Based on the following performance history: {history_json}, "
#     #       f"and model structures {model_descriptions['A']} {model_descriptions['B']}, {model_descriptions['C']}, determine which model should be trained next. "
#     #       f"Ensure that the training distribution among A, B, and C remains balanced, avoiding repeated training of the same model or oscillating between two models. "
#     #       f"Considering fairness and resource usage, return only the letter A, B, or C as the first character of your response.")
#     prompt = (
#         "Based on the following performance history: {history_json} for client {client_id} currently training {model_name}, "
#         "return exactly one letter (A, B, or C) to indicate which model to train next. The letter should be chosen based on the following priority order:\n"
#         "1. Choose the model with the lowest training count to maintain a balanced distribution.\n"
#         "2. Avoid training the same model consecutively.\n"
#         "3. Consider the model structures: {model_descriptions['A']}, {model_descriptions['B']}, {model_descriptions['C']}.\n\n"
#         "Return only the letter (A, B, or C) with no additional text, explanation, or punctuation."
#     )


#     messages = [{"role": "user", "content": prompt}]
#     print(f"History for client {client_id}: {history_json}")

#     try:
#         response = ollama_client.chat(model='llama3.3', messages=messages)
#         print(f"Response from Ollama: {response}")

#         if 'message' in response and 'content' in response['message']:
#             first_char = response['message']['content'].strip()[1].upper() 
#             if first_char in ['A', 'B', 'C']:
#                 print("Decision source: Ollama")
#                 #return first_char, "llama3"  # Use Ollama's decision if valid 现在是返回了两个值，如果只返回一个，用下面的注释掉的代码
#                 return first_char  # Use Ollama's decision if valid

#             else:
#                 print(f"Invalid decision from Ollama: {first_char}")
#     # except Exception as e:
#     #     print(f"Error calling Ollama: {e}")
#     # try:
#     #     response = ollama_client.chat(model='llama3', messages=messages)
#     #     print(f"Response from Ollama: {response}")

#     #     if 'message' in response and 'content' in response['message']:
#     #         content = response['message']['content'].strip()
#     #         # 使用正则表达式提取字母和理由
#     #         import re
#     #         match = re.match(r'^\[([ABC])\]\.\s*(.*)$', content)
#     #         if match:
#     #             selected_model = match.group(1)
#     #             reasoning = match.group(2).lower()
#     #             print("Decision source: Ollama")
#     #             # 检查返回的字母是否与理由中的模型一致
#     #             if selected_model.lower() in reasoning:
#     #                 return selected_model  # 返回选定的模型
#     #             else:
#     #                 print(f"Selected model and reasoning do not match: {content}")
#     #                 # 根据理由中的模型名称更新选定的模型
#     #                 for model in ['a', 'b', 'c']:
#     #                     if model in reasoning:
#     #                         return model.upper()
#     #                 # 如果无法匹配，则返回随机选择
#     #                 return random.choice(['A', 'B', 'C'])
#     #         else:
#     #             print(f"Invalid format from Ollama: {content}")
#     #             return random.choice(['A', 'B', 'C'])
#     except Exception as e:
#         print(f"Error calling Ollama: {e}")
#         return random.choice(['A', 'B', 'C'])



#     # Fallback to random choice if Ollama fails
#     print("Decision source: Random")
#     print("Ollama failed, falling back to random choice.")
#     #return random.choice(['A', 'B', 'C']), "Random"
#     return random.choice(['A', 'B', 'C']) # 现在是返回了两个值，如果只返回一个，用下面的注释掉的代码


# @app.route('/next_model_decision', methods=['POST'])
# def next_model_decision():
#     data = request.get_json()

#     if not data or 'prompt' not in data:
#         return jsonify({'error': 'Missing "prompt" in request data'}), 400

#     prompt = data['prompt']
#     print(f"Received prompt: {prompt}")

#     client_id = prompt['client_id']
#     model_name = prompt['model_name']

#     # Update client performance history
#     client_performance_data.setdefault(client_id, []).append(prompt)

#     # Always try to use Ollama when `use_ollama` is True
#     next_model = decide_model_for_next_training(client_id, model_name)
#     print(f"Model for next training: {next_model}")

#     return jsonify({'next_model': next_model})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
