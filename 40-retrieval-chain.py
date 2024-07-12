from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    # instantiate model
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}
    """)

    # create LLM Chain
    #chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})

    retriever_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retriever_chain

docs = get_documents_from_web('https://python.langchain.com/docs/expression_language/')
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input": "What is LCEL?",
})
print(response["context"])