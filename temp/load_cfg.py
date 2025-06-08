import yaml
import os


with open('config.yml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    

server_ip = config.get('server', str)
client1_ip = config.get('client1', str)
client2_ip = config.get('client2', str)
client3_ip = config.get('client3', str)



CLIENT_ID_value = os.getenv('CLIENT_ID')

if CLIENT_ID_value is not None:
    # print(f'The value of CLIENT_ID is: {CLIENT_ID_value}')
    current_ip = config.get(CLIENT_ID_value, str)
else:
    # print('Environment variable CLIENT_ID_value is not set.')
    current_ip = server_ip





if __name__ == '__main__':
    print(current_ip)

        