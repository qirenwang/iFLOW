# iFLOW: Intelligent Multi-Model Federated Learning Framework
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
python3 ss.py

# start the LLM agent
python3 flask_app.py

# start a client
python3 cc.py [CLIENT_ID] eg: CLIENT_ID = 1

# start another client
python3 cc.py [CLIENT_ID] eg: CLIENT_ID = 2/3/...
```
## Citation
Qiren Wang, Yongtao Yao, Nejib Ammar and Weisong Shi, "iFLOW: An Intelligent and Scalable Multi-Model Federated Learning Framework on the Wheels," *IEEE Transactions on Intelligent Transportation Systems ([TITS](https://ieee-itss.org/pub/t-its/))*, 2025. 
