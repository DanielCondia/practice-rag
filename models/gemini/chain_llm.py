from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from getpass import getpass

api_key = getpass("Enter your API key: ")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente util que traduce {input_language} a {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

# Encadenar la plantilla con el modelo de Google AI
chain = prompt | GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# Invocar la cadena con parametros especificos
response = chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "What are some of the pros and cons of Python as a programming language?",
    }
)

print(response)