# import time
# import ollama

# start_time = time.time()

# response = ollama.chat(model='llama3.2:latest', messages=[
#     {
#         'role': 'user',
#         'content': 'Why is the sky blue? Respond in a friendly manner and keep the answer to one line.',
#     },
# ])

# end_time = time.time()
# execution_time = end_time - start_time

# print(response['message']['content'])

# print(f"Execution time: {execution_time} seconds")