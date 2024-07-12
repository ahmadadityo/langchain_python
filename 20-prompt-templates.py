from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# instantiate model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chef. Create a unique recipe based on the follow main ingredient."),
        ("human", "{input}")
    ]
)

# create LLM Chain
chain = prompt | llm

response = chain.invoke({"input": "tomatoes"})
print(response)