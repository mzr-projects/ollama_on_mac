from ollama import Client

client = Client(trust_env=False)

# System prompt : sets the behavior of the model
# User prompt : the user's question
# Assistant response : the model's responses to the user's question

response = client.chat(
    model="gpt-oss:20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can answer questions about the capital of France under 100 words."},
        {"role": "user", "content": "Explain about the capital of France?"}
    ],
    # stream=True is used to stream the response from the model and send the response in small chunks
    stream=True
)

# flush=True is used to flush the output to the console immediately
# end="" is used to prevent the new line from being printed after each chunk
for chunk in response:
    print(chunk.message.content, end="", flush=True)

print()