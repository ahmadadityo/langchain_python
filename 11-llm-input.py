from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

# Fungsi untuk meminta input dari pengguna
def get_user_input():
    prompt = input("Lo: ")
    return prompt

# Meminta input dari pengguna
user_prompt = get_user_input()

# Inisialisasi ChatOpenAI dengan model GPT-3.5-turbo
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)

# Meminta respons berdasarkan input pengguna
response = llm.stream(user_prompt)

# Menampilkan respons secara streaming
print("T.A.I: ", end=" ")
for chunk in response:
    print(chunk.content, end="", flush=True)