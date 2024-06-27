import logging
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel
from ragstack_langchain.colbert import ColbertVectorStore

logging.basicConfig(level=logging.INFO)
logging.getLogger("cassandra").setLevel(logging.ERROR)
logging.getLogger("http").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

LLM_MODEL = "gpt-3.5-turbo"

batch_size = 640
chunk_size = 256
db_timeout = 1000


def get_lc_vector_store(keyspace: str, table_name: str) -> ColbertVectorStore:
    astra_token = os.getenv("ASTRA_DB_TOKEN")
    database_id = os.getenv("ASTRA_DB_ID")

    database = CassandraDatabase.from_astra(
        database_id=database_id,
        astra_token=astra_token,
        keyspace=keyspace,
        table_name=table_name,
        timeout=db_timeout,
    )
    embedding_model = ColbertEmbeddingModel(
        doc_maxlen=chunk_size, batch_size=batch_size
    )

    return ColbertVectorStore(
        database=database,
        embedding_model=embedding_model,
    )


def query_pipeline(keyspace: str, table_name: str, **kwargs):
    vector_store = get_lc_vector_store(keyspace=keyspace, table_name=table_name)
    llm = ChatOpenAI(model_name=LLM_MODEL)

    # build a prompt
    prompt_template = """
    Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
    Context: {context}
    Question: {question}
    Your answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {
            "context": vector_store.as_retriever(k=10),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
