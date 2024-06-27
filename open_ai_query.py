import os

from langchain_astradb import AstraDBVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"


def get_vector_store(collection_name: str):
    return AstraDBVectorStore(
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=collection_name,
        token=os.getenv("ASTRA_DB_TOKEN"),
        api_endpoint=os.getenv("ASTRA_DB_ENDPOINT"),
    )


def query_pipeline(collection_name: str, **kwargs):
    vector_store = get_vector_store(collection_name=collection_name)
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
            "context": vector_store.as_retriever(search_kwargs={"k": 5}),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
