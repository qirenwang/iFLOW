<!-- ----------------------------------------------------------- -->
<!--  iFLOW: Intelligent Multi-Model Federated Learning Framework -->
<!-- ----------------------------------------------------------- -->

<h1 align="center">
  iFLOW: Intelligent Multi-Model Federated Learning Framework
  <br/>
  <sub></sub>
</h1>

<div align="center">    
  <p style="font-size: 20px;">by 
    <a href="mailto:qirenw@udel.edu">Qiren Wang</a><sup>1</sup>,
    Yongtao Yao<sup>1</sup>, 
    Nejib Ammar<sup>2</sup>, 
    <a href="mailto:weisong@udel.edu">Weisong Shi</a><sup>1</sup>
  </p>
  <p>
    <sup>1</sup>University of Delaware, <sup>2</sup>InfoTech Lab, Toyota North America
  </p>
  
  <p style="font-size: 18px; font-weight: bold;">
    IEEE Transactions on Intelligent Transportation Systems
  </p>

[![DOI](https://img.shields.io/badge/DOI-10.1109/TITS.2025.3578586-blue.svg)](https://doi.org/10.1109/TITS.2025.3578586)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)
</div>

---

## 📜 Abstract
> The high mobility characteristics of connected vehicles present noteworthy difficulties in the domain of federated learning. Based on our understanding, current federated learning strategies do not tackle the challenge of continuously training multiple models for vehicles in constant motion, which are subject to variable network conditions and changing environments. In response to this challenge, we have created and implemented iFLOW, a versatile and intelligent multi-model federated learning infrastructure specifically designed for highly mobile-connected vehicles. iFLOW addresses these challenges by integrating four key aspects: (1) a strategically devised model allocation algorithm that dynamically selects vehicle computing units for distinct model training tasks, optimizing for both resource efficiency and performance; (2) a dynamic client vehicle joining mechanism that ensures smooth participation of vehicles, even in the face of signal loss or weak connectivity, mitigating disruptions in the training process; (3) integration of a large language model (Llama3.3 70B) as an intelligent arbiter for decision-making within the framework, enhancing adaptability and robustness; and (4) real-world deployment and testing on distributed vehicular devices to validate the approach. The experimental evaluation demonstrates that iFLOW allows multiple models to train asynchronously and outperform centralized training. These results affirm the effectiveness of iFLOW in practical, real-world scenarios involving highly mobile vehicular networks.

---

## 🗺️ Table of Contents
1. [Setup](#setup)
2. [How to Use](#how-to-use)
3. [Citation](#citation)

---

## Setup

### 1. Setup Torch and TorchVision
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install paho-mqtt==2.0.0 (Optional)
pip3 install psutil
```

### 2. Mosquitto Setup (Optional)
```bash
sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa
sudo apt-get update
sudo apt-get install mosquitto
mosquitto
```

---

## How to Use

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

---

## Citation
```bibtex
@article{wang2025iflow,
  title={iFLOW: An Intelligent and Scalable Multi-Model Federated Learning Framework on the Wheels},
  author={Wang, Qiren and Yao, Yongtao and Ammar, Nejib and Shi, Weisong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE},
  doi={10.1109/TITS.2025.3578586}
}
```

---

<p align="center">♥ Please reach out to <a href="mailto:qirenw@udel.edu">qirenw@udel.edu</a> or <a href="mailto:weisong@udel.edu">weisong@udel.edu</a> for any inquiries ♥</p>
