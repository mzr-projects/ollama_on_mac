from ollama import Client

client = Client(trust_env=False)

response = client.chat(
    model="gpt-oss:20b",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
)

print(response["message"]["content"])