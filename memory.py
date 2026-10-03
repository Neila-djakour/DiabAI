from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)
checkpointer = InMemorySaver()
agent = create_agent(
    model=model,
    system_prompt="You are a helpful assistant.",
    checkpointer=checkpointer
)


result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My favorite color is green."
            }
        ]
    },
    config={
        "configurable": {
            "thread_id": "user-1"
        }
    }
)

result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is my favorite color?"
            }
        ]
    },
    config={
        "configurable": {
            "thread_id": "user-2"
        }
    }
)

print(result2["messages"][-1].content)
