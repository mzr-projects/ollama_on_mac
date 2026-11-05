from ollama import Client

client = Client(trust_env=False)

# System prompt : sets the behavior of the model
# User prompt : the user's question
# Assistant response : the model's responses to the user's question

history = [{
    "role": "system",
    "content": "You are a helpful assistant, Be concise and to the point."
}]

print("Chat with AI (type quit exit or bye to end)")
print("-" * 40)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bye!")
        break

    history.append({"role": "user", "content": user_input})

    stream = client.chat(
        model="gpt-oss:20b",
        messages=history,
        stream=True
    )

    print("AI: ", end="", flush=True)
    response_text = ""
    for chunk in stream:
        piece = chunk.message.content or ""
        print(chunk.message.content, end="", flush=True)
        response_text += str(piece)
    print()

    history.append({"role": "assistant", "content":response_text})