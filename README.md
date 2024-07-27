# Intelligent Multi-Model Federated Learning Framework
## 1. Setup Torch and TorchVision
```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip3 install paho-mqtt==2.0.0
    pip3 install psutil
```

## 2. Mosquitto Setup
```bash
    sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa
    sudo apt-get update
    sudo apt-get install mosquitto
    mosquitto
```
## 3. How to use
```bash
# start the server
python3 main_server.py

# start a client
python3 main_client.py client1

# start another client
python3 main_client.py client2
```
