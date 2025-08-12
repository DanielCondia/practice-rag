import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Cargar variables de entorno
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
google_api_key = os.getenv("GOOGLE_API_KEY")

# 1. Crear generador de embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id='BAAI/bge-large-en-v1.5',
    huggingfacehub_api_token=hf_token
)

# 2. Simulamos o cargamos documentos
docs = [
    Document(page_content='La capital de Colombia es Bogotá.', metadata={'fuente': 'doc1'}),
    Document(page_content='El café de Colombia es reconocido internacionalmente por su calidad', metadata={'fuente': 'doc2'}),
]

# 3. Chunking si las fuentes son grandes
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

print('split_docs: ', split_docs)

# 4. Crear el vector store
vectorstore = Chroma.from_documents(
    split_docs,
    embeddings,
    persist_directory='./chrome_store'
)

# Debuggiar base de datos
docs = vectorstore.get()
print('docs: ', docs)

# print('vectorstore: ', vectorstore.as_retriever(search_kwargs={"k": 2}))

# 5. Cargar modelo
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=google_api_key,
    temperature=0.5
)

# 6. Cadena RAG con langchain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# 7. Consulta
query = 'El café de Colombia no es reconocido internacionalmente por su calidad?'
result = qa_chain.invoke(query)

print('Respuesta:', result['result'])
# print('\n Fuentes:')
# for doc in docs:
#     fuente = doc.metadata.get("fuente", "desconocido")
#     print("-", fuente, ":", doc.page_content)

