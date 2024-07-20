import logging
from typing import Any, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class ChatGLM(LLM):
    """ChatGLM LLM service.

    Example:
        .. code-block:: python

            from langchain_community.llms import ChatGLM
            endpoint_url = (
                "http://127.0.0.1:8000"
            )
            ChatGLM_llm = ChatGLM(
                endpoint_url=endpoint_url
            )
    """

    endpoint_url: str = "http://127.0.0.1:8000/"
    """Endpoint URL to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""
    max_token: int = 20000
    """Max token allowed to pass to the model."""
    temperature: float = 0.1
    """LLM model temperature from 0 to 10."""
    history: List[List] = []
    """History of the conversation"""
    top_p: float = 0.7
    """Top P for nucleus sampling from 0 to 1"""
    with_history: bool = False
    """Whether to use history or not"""

    @property
    def _llm_type(self) -> str:
        return "chat_glm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.endpoint_url},
            **{"model_kwargs": _model_kwargs},
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        messages = [
        {
            "role": "user", "content": "What's the Celsius temperature in San Francisco?"
        }]
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

        response = requests.post(self.endpoint_url , headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            # print("Result Type:", type(result))#<class 'dict'>
            # print("Result Content:", result['choices'][0]['message']['content'])  # 确保这一行是正确的路径
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")
            

import logging
from langchain.llms import BaseLLM
import json
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.runnables import RunnableSequence
# from langchain_community.llms import ChatGLM

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

endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"

# 创建客户端实例
llm = ChatGLM(endpoint_url=endpoint_url)
# print("-------1-----")

# # 设置日志
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# 创建并使用prompt模板
template = """Question: {question}
Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

# print(llm._call(prompt))
chains = LLMChain(prompt=prompt, llm=llm)
question = "请你介绍一下数据库有哪几种类型"#我上次问的什么问题


# 使用 `invoke` 方法代替 `run` 方法
try:
    response = chains.invoke(question)
    print("---------2--------")
    print(response)
    print("---------3--------")
except Exception as e:
    logger.error(f"Error while invoking the chain: {e}")