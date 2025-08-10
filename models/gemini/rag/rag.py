
from langchain_google_genai.google_vector_store import GoogleVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from getpass import getpass

# Obtener la API key de Google AI
api_key = getpass("Enter your Google AI API key: ")

# Configurar el almacenamiento vectorial con la API key
corpus_store = GoogleVectorStore.create_corpus(
    display_name="My Corpus",
)

# Create a document under that corpus
document_store = GoogleVectorStore.create_document(
    corpus_id=corpus_store.corpus_id,
    display_name="My Document"
)

# Load and upload documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
for file in DirectoryLoader(path="'../../../info_pdfs/preguntas_frecuentes.pdf'").load():
    chunks = text_splitter.split_documents([file])
    document_store.add_documents(chunks)

# Query the document corpus
aqa = corpus_store.as_aqa(
    model="gemini-2.5-flash",
    google_api_key=api_key)
response = aqa.invoke("Que horarios tiene la seccional de sogamoso?")

print("Answer:", response.answer)
print("Passages:", response.attributed_passages)
print("Answerable probability:", response.answerable_probability)