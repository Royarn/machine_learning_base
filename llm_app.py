from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.tongyi import Tongyi
import dashscope

# 直接设置 API Key
dashscope.api_key = 'sk-286924b330de4518a3ba644b00066a2f'

# 调试信息
print(f"dashscope.api_key: {dashscope.api_key}")

llm = Tongyi(dashscope_api_key=dashscope.api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是世界级的技术专家"),
    ("user", "{input}")])

# pretty result
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

result = chain.invoke({"input": "帮我写一个关于AI的技术文章,200个字 "})

print(result)