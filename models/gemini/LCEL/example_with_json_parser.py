from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from getpass import getpass

api_key = getpass("Enter your API key: ")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answer the following question: {question}",
)

json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | llm | json_parser

print(list(json_chain.stream({"question": "What are some of the pros and cons of Python as a programming language?"})))