from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

agent = create_agent(
    model=model,
    system_prompt="""
You are DiabAI, a simple diabetes education assistant.

Explain basic diabetes concepts clearly and simply.
Provide educational information only.
Do not diagnose patients or prescribe treatments.
For personal medical decisions, recommend consulting a qualified healthcare professional.
"""
)

print("DIABAI AGENT CREATED")

question = input("You: ")

result = agent.invoke({
    "messages": [
        {"role": "user", "content": question}
    ]
})

print("DiabAI:", result["messages"][-1].content)





