import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langsmith import Client
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
google_api_key = os.getenv("GOOGLE_API_KEY")

file_path = ('../../../info_pdfs/preguntas_frecuentes.pdf')
loader = PyPDFLoader(file_path)
documents = loader.load_and_split()
#print(documents)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", ".", " ", ""]
)

docs_chunked = text_splitter.split_documents(documents)

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id='BAAI/bge-m3',
    huggingfacehub_api_token=hf_token
)

persist_directory = './chroma_db_pdf'
vectordb = Chroma.from_documents(
    docs_chunked,
    embeddings,
    persist_directory=persist_directory
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=google_api_key,
    temperature=0.5
)
ls_token = os.getenv("LANGSMITH_API_KEY")
client = Client(api_key=ls_token)
prompt = client.pull_prompt('rlm/rag-prompt', include_model=True)

retriever = vectordb.as_retriever(search_type='similarity', search_kwargs={'k': 5})

qa_chain = (
    {
        'context': vectordb.as_retriever(),
        'question': RunnablePassthrough()
    }
    | prompt
    | llm
)

query = 'Que horarios de atención tiene la sede de Sogamoso?'
print(qa_chain.invoke(query))

