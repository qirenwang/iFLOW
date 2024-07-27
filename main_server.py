# parameter_server.py
from ParameterServerMQTT import ParameterServerMQTT

if __name__ == "__main__":
    # Instantiate and run the parameter server
    server = ParameterServerMQTT()
    try:
        #Keep running and listen for client messages
        while True:
            pass
    except KeyboardInterrupt:
        print("Parameter server is shutting down.")