# parameter_server.py
import time
from ParameterServerMQTT import ParameterServerMQTT

if __name__ == "__main__":
    # Instantiate and run the parameter server
    server = ParameterServerMQTT()
    try:
        #Keep running and listen for client messages
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Parameter server is shutting down.")