from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import pandas as pd
from pymilvus import FieldSchema, DataType, CollectionSchema, Collection, connections
from zhipuai import ZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings

# Load environment variables from .env file
load_dotenv("../.env")

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.environ.get("OPENAI_API_BASE")

# Initialize Flask application
app = Flask(__name__)

# Milvus connection setup
connections.connect(
    alias='default',
    uri=os.environ.get("CLUSTER_ENDPOINT"),
    token=os.environ.get("TOKEN"),
)

# Vector store configuration
vectorstore = Milvus(
    embedding_function=ZhipuAIEmbeddings(model="embedding-2"),
    collection_name="big_create_demo",
    connection_args={
        "uri": os.environ.get("CLUSTER_ENDPOINT"),
        "token": os.environ.get("TOKEN"),
        "secure": True,
    },
    primary_field="id",
    text_field="text",
    vector_field="text_vector",
)

# Retrieval configuration
retriever = vectorstore.as_retriever(search_kwargs=dict(k=5))

# Chatbot setup
llm = ChatOpenAI()

# Prompt template for responses
template = '''你是回答问题的助理。使用以下检索到的上下文来回答问题，在回复答案时请携带原文（如有）。如果你不知道答案，就说你不知道。
上下文: {context} 
问题: {question} 
回答:'''

prompt = PromptTemplate.from_template(template=template)


# Define routes
@app.route('/query', methods=['POST'])
def query():
    try:
        # Receive JSON data with question
        data = request.get_json()
        question = data['question']

        # Invoke retrieval and language model chain
        res = llm.invoke({"context": retriever, "question": question} | prompt | llm)

        return jsonify({"answer": res})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
