from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位人工智能助手，你的名字是{name}"),
    ("human", "你好"),
    ("ai", "我很好，谢谢"),
    ("human", "{user_input}")
])

message = prompt.format_messages(name="Bob", user_input="你的名字叫什么")

print(message)