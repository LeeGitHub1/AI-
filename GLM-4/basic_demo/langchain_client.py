from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = 'http://101.7.149.249:8100/v1/'

endpoint_url = "http://101.7.149.249:8100/v1/"
llm = ChatOpenAI()

template = """Question: {question}
Answer: """

prompt = PromptTemplate.from_template(template=template)

chains = prompt | llm | parser
# question = "请你介绍一下数据库有哪几种类型"#我上次问的什么问题

messages = [
    {
        "role": "system",
        "content": "请在每个段落开头都带上“亚麻跌~”三个字，放在开头。",
    },
    {
        "role": "user",
        "content": "请你介绍一下数据库有哪几种类型"
    }
]

response = chains.stream(input=messages)
for text in response:
    print(text, end='', flush=True)