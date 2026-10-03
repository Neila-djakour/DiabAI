from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=1
)

response = model.invoke([
    {
        "role": "system",
        "content": "Answer medical questions in exactly one short sentence."
    },
    {
        "role": "user",
        "content": "What is insulin?"
    },
    {
        "role": "assistant",
        "content": "Insulin is a hormone that helps glucose move from the blood into cells."
    },
    {
        "role": "user",
        "content": "What is diabetes?"
    }
])


print(response)
