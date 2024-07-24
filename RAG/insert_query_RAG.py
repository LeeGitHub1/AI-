from flask import Flask, request, jsonify
import os
from langchain_community.embeddings import ZhipuAIEmbeddings
from dotenv import load_dotenv
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import pandas as pd

app = Flask(__name__)

# Load environment variables
load_dotenv("../.env")

# Initialize Zhipu AI embeddings
embd_zhipu = []


def get_zhipu_embeddings_docs(texts):
    embeddings = ZhipuAIEmbeddings(
        api_key=os.environ.get("ZHIPUAI_API_KEY")
    )
    return embeddings.embed_documents(texts)


# 用于查询
def get_zhipu_embeddings_queries(texts):
    embeddings = ZhipuAIEmbeddings(
        api_key=os.environ.get("ZHIPU_API_KEY")
    )
    return embeddings.embed_documents(texts)


# Connect to Milvus
connections.connect(
    alias='default',
    uri=os.environ.get("CLUSTER_ENDPOINT"),
    token=os.environ.get("TOKEN")
)


def get_schema():
    field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    field2 = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
    field4 = FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
    schema = CollectionSchema(fields=[field1, field2, field4])
    return schema


def recreate_collection(collection_name):
    utility.drop_collection(collection_name=collection_name)
    schema = get_schema()
    collection = Collection(name=collection_name, schema=schema)
    index_params = {
        "index_type": "AUTOINDEX",
        "metric_type": "L2",
        "params": {}
    }
    collection.create_index(
        field_name="text_vector",
        index_params=index_params,
        index_name='vector_idx'
    )
    collection.load()
    return collection


def get_collection(collection_name):
    return Collection(name=collection_name)


def insert_data(collection, df):
    vectors = df['embedding'].tolist()
    data = [
        {"text": f"{text}", "text_vector": vector} for text, vector in zip(df['text'], vectors)
    ]
    collection.insert(data)


def search(collection, query_embedding, top_k=5):
    search_params = {
        "metric_type": "L2",
        "params": {"level": 2}
    }
    results = collection.search(
        query_embedding,
        anns_field="text_vector",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )
    return results


@app.route('/insert', methods=['POST'])
def insert():
    data = request.json
    texts = data.get('texts', [])
    embeddings = get_zhipu_embeddings_docs(texts)
    collection_name = "big_create_demo"
    collection = get_collection(collection_name)
    df = pd.DataFrame({'text': texts, 'embedding': embeddings})
    insert_data(collection, df)
    collection.flush()
    return jsonify({'status': 'success', 'message': 'Data inserted successfully'})


@app.route('/search', methods=['POST'])
def search_api():
    data = request.json
    query_text = data.get('query_text', '')
    top_k = data.get('top_k', 5)
    collection_name = "big_create_demo"
    collection = get_collection(collection_name)
    query_embedding = get_zhipu_embeddings_queries([query_text])
    results = search(collection, query_embedding[0], top_k=top_k)
    texts = [i.entity.get('text') for i in results[0]]
    return jsonify({'status': 'success', 'results': texts})


if __name__ == '__main__':
    app.run(debug=True)
