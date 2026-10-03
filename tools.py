from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

load_dotenv()


@tool
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    "Calculate a person's Body Mass Index (BMI) using weight in kilograms and height in meters."
    return weight_kg / (height_m * height_m)


model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

agent = create_agent(
    model=model,
    tools=[calculate_bmi],
    system_prompt="""
You are DiabAI, a diabetes education assistant.
Use the BMI calculation tool when someone asks you to calculate BMI.
Provide educational information only.
Do not diagnose patients or prescribe treatments.
"""
)

print("DIABAI WITH TOOL CREATED")

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Calculate my BMI if I weigh 80 kg and I am 1.60 m tall."
        }
    ]
})

for message in result["messages"]:
    print("-----")
    print(message)
