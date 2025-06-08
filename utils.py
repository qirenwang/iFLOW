import requests


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

def decide_model_for_next_training(prompt):
    # Flask server's URL
    url = "http://localhost:5000/next_model_decision"

    # JSON data to send
    payload = {
        "prompt": prompt
    }


    # Send POST request
    response = requests.post(url, json=payload)

    # Check the response status code and get the result
    if response.status_code == 200:
        result = response.json()  # Parse the returned JSON data
        print("Server response:", result)
        return result['next_model']
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Error message:", response.text)
        return None
