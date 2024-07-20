import logging
from langchain.llms import BaseLLM
import json
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables.base import RunnableSequence
from langchain_community.llms import ChatGLM

logger = logging.getLogger(__name__)


messages = [
        {
            "role": "user", "content": "What's the Celsius temperature in San Francisco?"
        },

        # Give Observations
        # {
        #     "role": "assistant",
        #         "content": None,
        #         "function_call": None,
        #         "tool_calls": [
        #             {
        #                 "id": "call_1717912616815",
        #                 "function": {
        #                     "name": "get_current_weather",
        #                     "arguments": "{\"location\": \"San Francisco, CA\", \"format\": \"celsius\"}"
        #                 },
        #      r           "type": "function"
        #             }
        #         ]
        # },
        # {
        #     "tool_call_id": "call_1717912616815",
        #     "role": "tool",
        #     "name": "get_current_weather",
        #     "content": "23°C",
        # }
    ]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
]

class ChatGLM(BaseLLM):
    endpoint_url: str = "http://127.0.0.1:8000/"
    model_kwargs: dict = None

    @property
    def _llm_type(self) -> str:
        """Identify the type of the LLM."""
        return "chat_glm"

    def _generate(self, prompt, **kwargs):
        """Generate text using the configured LLM model."""
        headers = {"Content-Type": "application/json"}
        # 仅使用 messages 字段，移除 message 字段，确保 content 是单一字符串
        payload = {
            "model": "glm-4",
            "messages": messages,
            "temperature": self.temperature,
            "max_length": self.max_token,
            "top_p": self.top_p,
        }
        if self.history:
            payload["history"] = self.history

        response = requests.post(self.endpoint_url + "v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            print("Result Type:", type(result))#<class 'dict'>
            print("Result Content:", result['choices'][0]['message']['content'])  # 确保这一行是正确的路径
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")


# 创建客户端实例
llm = ChatGLM(temperature=0.1)

# # 设置日志
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# 创建并使用prompt模板
template = """Question: {question}
Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])


# chains = LLMChain(prompt=prompt, llm=llm)
# question = "请你介绍一下数据库有哪几种类型"#我上次问的什么问题

# print(1)
# # 使用 `invoke` 方法代替 `run` 方法
# try:
#     response = chains.invoke(question)
#     print("11111111111111")
#     print(response)
# except Exception as e:
#     logger.error(f"Error while invoking the chain: {e}")

runnable = RunnableSequence(prompt | llm)
question = "请你介绍一下数据库有哪几种类型"

try:
    response = runnable.invoke(question)
    print(response)
except Exception as e:
    logger.error(f"Error while running the sequence: {e}")