import os
import logging

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.astradb import AstraDB
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI

from typing import List


def get_vector_store(chunk_size:int):
    return AstraDB(
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        collection_name=f"chunk_size_{chunk_size}",
        token=os.getenv("ASTRA_DB_TOKEN"),
        api_endpoint=os.getenv("ASTRA_DB_ENDPOINT"),
    )


def ingest(file_paths: List[str], chunk_size:int, **kwargs):
    vector_store =  get_vector_store(chunk_size=chunk_size)

    chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))
    logging.info(f"Using chunk_overlap: {chunk_overlap} for chunk_size: {chunk_size}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt3.5",
        chunk_size=chunk_size,
        chunk_overlap=50,
    )

    docs = UnstructuredFileLoader(file_path=file_paths, mode="single", strategy="fast")
    split_docs = text_splitter.split_documents(docs)
    vector_store.add_documents(split_docs)


def query_pipeline(k: int, chunk_size: int, **kwargs):
    vector_store =  get_vector_store(chunk_size=chunk_size)
    llm = OpenAI(model_name="gpt-3.5-turbo")

    # build a prompt
    prompt_template = """
    Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
    Context: {context}
    Question: {question}
    Your answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": vector_store.as_retriever(search_kwargs={'k': k}), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain