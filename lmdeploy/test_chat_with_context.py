# Let's chat with AI with context
from openai import OpenAI

# Load the API key
client = OpenAI(
    # LM API base url
    base_url="http://localhost:23333/v1",
    # If you have not configured the environment variable, replace the line below with the following: api_key="sk-xxx"
    api_key="sk-xxx")

# The context of the conversation
chat_history = []
while True:
    prompt = input("User: ")
    chat_history.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="/root/llm/Qwen/Qwen2.5-0.5B-Instruct",
        messages=chat_history,
        stream=False,
    )
    print("AI: ", response.choices[0].message.content)