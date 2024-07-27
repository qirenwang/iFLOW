# import ollama

# stream = ollama.chat(
#     model='llama3',
#     messages=[{'role': 'user', 'content': 'Why is the result of 2/2, don not explain, just give me answer'}],
#     stream=True,
# )

# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)
from ollama import Client
client = Client(host='http://10.0.0.31:11434')
stream = client.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': 'What is the PDE 偏微分?',
  }],
stream=True,)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)