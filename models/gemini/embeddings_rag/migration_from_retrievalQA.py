import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langsmith import Client

# See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=google_api_key,
    temperature=0.5
)
docs = [
    Document(page_content='La capital de Colombia es Bogotá.', metadata={'fuente': 'doc1'}),
    Document(page_content='El café de Colombia es reconocido internacionalmente por su calidad', metadata={'fuente': 'doc2'}),
]
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id='BAAI/bge-large-en-v1.5',
    huggingfacehub_api_token=hf_token
)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    split_docs,
    embeddings,
    persist_directory='./chrome_store'
)

ls_token = os.getenv("LANGSMITH_API_KEY")
client = Client(api_key=ls_token)
prompt = hub.pull('rlm/rag-prompt')
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        'context': vectorstore.as_retriever() | format_docs,
        'question': RunnablePassthrough()
    }
    | prompt | llm | StrOutputParser()
)

print(qa_chain.invoke("Cual es la capital de Estados Unidos?"))