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
)

print(response["message"]["content"])