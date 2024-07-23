import logging
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from langchain_community.llms import ChatGLM
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Milvus
import clip
# from prompt_encoder import zhipu_get_embedding
from langchain_community.embeddings import ZhipuAIEmbeddings
import os
# 设置日志记录
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ['ZHIPUAI_API_KEY'] = '1de80e45bad68a515504444b57330cb9.qcNmqzoX22Dwtbfh'
# 指定ChatGLM 服务的端点 URL
endpoint_url = "http://127.0.0.1:8000/chat"

llm = ChatGLM(
    endpoint_url=endpoint_url
)

# model, preprocess = clip.load("ViT-B/32")


vectorstore = Milvus(
    embedding_function=ZhipuAIEmbeddings(model="embedding-2"),
    collection_name="search_collection",
    connection_args={
        "uri": "https://in03-94eca040160cf1c.api.gcp-us-west1.zillizcloud.com",
        "token": "1e348be72761c59c340e5b99a24aae021d0a0051239b15b82194e1f5c2612f27b12faf7c72b3bda0009bce37660363f2513e4466",
        # API key, for serverless clusters which can be used as replacements for user and password
        "secure": True,
    },
    primary_field="id",
    text_field="description",
    vector_field="embedding",
)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=5))

template = """Question: {question}
Context：{context}
Answer: """
prompt = PromptTemplate(template=template, input_variables=["question", "context"])
print(prompt)


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)