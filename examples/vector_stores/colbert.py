import os
import time
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from ragstack_colbert import (
    CassandraDatabase,
    Chunk,
    ColbertEmbeddingModel,
    ColbertVectorStore,
)
from ragstack_langchain.colbert import ColbertVectorStore as LangChainColbertVectorStore
from transformers import BertTokenizer

from ssl import SSLContext, PROTOCOL_TLS, CERT_REQUIRED

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import AddressTranslator
import cassio

LLM_MODEL = "gpt-3.5-turbo"

batch_size = 640

keyspace = "default_namespace"

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("cassandra").setLevel(logging.ERROR)



def get_embedding_model(chunk_size: int) -> ColbertEmbeddingModel:
    return ColbertEmbeddingModel(doc_maxlen=chunk_size, batch_size=batch_size)


def get_database(chunk_size: int) -> CassandraDatabase:
    table_name = f"colbert_chunk_size_{chunk_size}"

    class StaticTranslator(AddressTranslator):
        """
        Returns the endpoint with translation
        """
        def translate(self, addr):
            return fos.getenv("CASSANDRA_CONTACT_POINT")

    address_translator = StaticTranslator()
    auth_provider = PlainTextAuthProvider(username=os.getenv("CASSANDRA_USERNAME"), password=os.getenv("CASSANDRA_PASSWORD"))

    ssl_context = SSLContext(PROTOCOL_TLS)
    ssl_context.load_verify_locations(os.getenv("SSL_CERT_FILE"))
    ssl_context.verify_mode = CERT_REQUIRED

    cluster = Cluster([os.getenv("CASSANDRA_CONTACT_POINT")], ssl_context=ssl_context, port={os.environ["CASSANDRA_PORT"]}, auth_provider=auth_provider, connect_timeout=30, address_translator=address_translator)
    session = cluster.connect()

    cassio.init(session=session, keyspace="default_namespace")

    database = CassandraDatabase.from_session(
        session= session,
        keyspace="default_namespace",
        table_name=table_name,
    )
    return database


def get_lc_vector_store(chunk_size: int) -> LangChainColbertVectorStore:
    database = get_database(chunk_size=chunk_size)
    embedding_model = get_embedding_model(chunk_size=chunk_size)

    return LangChainColbertVectorStore(
        database=database,
        embedding_model=embedding_model,
    )


def get_vector_store(chunk_size: int) -> ColbertVectorStore:
    database = get_database(chunk_size=chunk_size)
    return ColbertVectorStore(database=database)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def len_function(text: str) -> int:
    return len(tokenizer.tokenize(text))


async def ingest(file_path: str, chunk_size: int, **kwargs):
    doc_id = Path(file_path).name

    chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))

    start = time.time()
    docs = UnstructuredFileLoader(
        file_path=file_path, mode="single", strategy="fast"
    ).load()
    duration = time.time() - start
    print(f"It took {duration} seconds to load and parse the document")

    # confirm only one document returned per file
    assert len(docs) == 1

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len_function,
    )

    start = time.time()
    chunked_docs = text_splitter.split_documents(docs)
    duration = time.time() - start
    print(
        f"It took {duration} seconds to split the document into {len(chunked_docs)} chunks"
    )

    texts = [doc.page_content for doc in chunked_docs]
    start = time.time()
    embeddings = get_embedding_model(chunk_size=chunk_size).embed_texts(texts=texts)
    duration = time.time() - start
    print(f"It took {duration} seconds to embed {len(chunked_docs)} chunks")

    colbert_vector_store = get_vector_store(chunk_size=chunk_size)

    await colbert_vector_store.adelete_chunks(doc_ids=[doc_id])

    chunks: List[Chunk] = []
    for i, doc in enumerate(chunked_docs):
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=i,
                text=doc.page_content,
                metadata={} if doc.metadata is None else doc.metadata,
                embedding=embeddings[i],
            )
        )

    start = time.time()
    await colbert_vector_store.aadd_chunks(chunks=chunks, concurrent_inserts=100)
    duration = time.time() - start
    print(
        f"It took {duration} seconds to insert {len(chunked_docs)} chunks into AstraDB"
    )


def query_pipeline(k: int, chunk_size: int, **kwargs):
    vector_store = get_lc_vector_store(chunk_size=chunk_size)
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
            "context": vector_store.as_retriever(search_kwargs={"k": k}),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain