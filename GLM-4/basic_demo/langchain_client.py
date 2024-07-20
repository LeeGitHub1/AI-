import logging
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# # 设置日志记录
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 指定ChatGLM 服务的端点 URL
endpoint_url = "http://127.0.0.1:8000/v1/"

llm = ChatOpenAI(
    model = "GLM-4-9B",
    openai_api_key="EMPTY",
    openai_api_base=endpoint_url,
    stream_options=True,
    stream_usage=True
    # streaming = True
    # history=[["用python写一下决策树解决实际问题的程序,给我完整的代码"]],
    # with_history=True
)

template = """Question: {question}
Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

chains = LLMChain(prompt=prompt, llm=llm)
# question = "请你介绍一下数据库有哪几种类型"#我上次问的什么问题

messages = [
    {
        "role": "system",
        "content": "请在你输出的时候都带上“啦啦啦”三个字，放在开头。",
    },
    {
        "role": "user",
        "content": "请你介绍一下数据库有哪几种类型"
    }
]
# # 使用 `invoke` 方法代替 `run` 方法
# try:
#     response = chains.invoke(input=messages)
#     print(response['text'])
# except Exception as e:
#     logger.error(f"Error while invoking the chain: {e}")

# response = chains.invoke(model="glm-4",input=messages)
response = chains.invoke(input=messages)
# for chunk in response['text']:
#     print(chunk)
print(response['text'])