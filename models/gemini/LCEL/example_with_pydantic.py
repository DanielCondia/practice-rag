from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, model_validator

import getpass

api_key = getpass.getpass("Enter your API key: ")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# define your desired data structure
class Joke(BaseModel):
    setup: str = Field(description='question to set up a joke')
    punchline: str = Field(description='answer to resolve a joke')

    @model_validator(mode='before')
    @classmethod
    def question_ens_with_question_mark(cls, values: dict) -> dict:
        setup = values.get('setup')
        if setup and setup[-1] != '?':
            raise ValueError('setup must end with a question mark')
        return values

# set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query. \n {format_instructions}\{query}\n",
    input_variables=['query'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# prompt_and_model = prompt | llm
# output = prompt_and_model.invoke({'query': 'Tell me a joke'})
# response = parser.invoke(output)

# print(response)

# with chain
chain = prompt | llm | parser
output = chain.invoke({'query': 'Tell me a joke'})

print(output)