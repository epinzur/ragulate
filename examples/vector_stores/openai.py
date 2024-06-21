import os
from astrapy.db import AstraDB

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
import logging

from langchain_community.vectorstores import Neo4jVector, ElasticsearchStore, Redis

import weaviate
from langchain_weaviate import WeaviateVectorStore


from langchain_astradb import AstraDBVectorStore
from langchain_community.document_loaders import UnstructuredFileLoader

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo-0125" #"gpt-4o-2024-05-13"

embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def get_store_name(chunk_size: int):
    return f"ragulate_chunk_{chunk_size}"

def get_astra_vector_store(chunk_size: int):
    astra_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

    db_client = AstraDB(api_endpoint=astra_endpoint, astra_token=astra_token, api_path="/", api_version="v1", token=astra_token, namespace="default_namespace")
    async_db_client = db_client.to_async()

    vstore = AstraDBVectorStore(
            collection_name=get_store_name(chunk_size=chunk_size),
            embedding=embedding,
            astra_db_client=db_client,
            async_astra_db_client=async_db_client,
        )
    return vstore

def get_neo4j_vector_store(chunk_size: int):
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    index_name = get_store_name(chunk_size=chunk_size)

    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vstore = Neo4jVector(username=neo4j_user,
        password=neo4j_password,
        url=neo4j_uri,
        index_name=index_name,
        embedding=embedding)

    def check_index_exists():
        query = f"CALL db.indexes() YIELD name WHERE name = '{index_name}' RETURN name"
        with vstore._driver.session() as session:
            result = session.run(query)
            return result.single() is not None

    if not check_index_exists(vector_store=vstore, index_name=index_name):
        vstore.create_new_index()

    return vstore

def get_elastic_vector_store(chunk_size: int):
    es_cloud_id = os.getenv("ELASTIC_CLOUD_ID")
    es_api_key = os.getenv("ELASTIC_API_KEY")

    vstore = ElasticsearchStore(
        es_cloud_id=es_cloud_id,
        es_api_key=es_api_key,
        index_name=get_store_name(chunk_size=chunk_size),
        embedding=embedding
    )
    return vstore

def get_redis_vector_store(chunk_size: int):
    redis_url = os.getenv("REDIS_URL")
    vstore = Redis(
        redis_url=redis_url,
        index_name=get_store_name(chunk_size=chunk_size),
        embedding=embedding
    )

    return vstore

def get_weaviate_vector_store(chunk_size: int):
    weaviate_host = os.getenv("WEAVIATE_HOST")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    client = weaviate.Client(
        url=weaviate_host,
        auth_client_secret=weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
    )

    vstore = WeaviateVectorStore(
        collection_name=get_store_name(chunk_size=chunk_size),
        embedding=embedding,
        weaviate_client=client
    )
    return vstore


def get_vector_store(vector_store: str, chunk_size: int):
    vector_store = vector_store.lower()
    if vector_store == "astradb":
        return get_astra_vector_store(chunk_size=chunk_size)
    elif vector_store == "elastic":
        return get_elastic_vector_store(chunk_size=chunk_size)
    elif vector_store == "neo4j":
        return get_neo4j_vector_store(chunk_size=chunk_size)
    elif vector_store == "redis":
        return get_redis_vector_store(chunk_size=chunk_size)
    elif vector_store == "weaviate":
        return get_weaviate_vector_store(chunk_size=chunk_size)
    else:
        raise ValueError(f"Unknown vector store type: {vector_store}")


def ingest(file_path: str, chunk_size: int, vector_store: str, **kwargs):
    vstore = get_vector_store(chunk_size=chunk_size, vector_store=vector_store)

    chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))
    logging.info(f"Using chunk_overlap: {{chunk_overlap}} for chunk_size: {{chunk_size}}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=EMBEDDING_MODEL,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs = UnstructuredFileLoader(
        file_path=file_path, mode="single", strategy="fast"
    ).load()
    split_docs = text_splitter.split_documents(docs)
    vstore.add_documents(split_docs)

def query_pipeline(k: int, chunk_size: int,  vector_store: str, **kwargs):
    vstore = get_vector_store(chunk_size=chunk_size, vector_store=vector_store)
    llm = ChatOpenAI(model_name=LLM_MODEL)

    # build a prompt
    prompt_template = """
    Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
    Context: {{context}}
    Question: {{question}}
    Your answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {{
            "context": vstore.as_retriever(search_kwargs={{"k": k}}),
            "question": RunnablePassthrough(),
        }}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
