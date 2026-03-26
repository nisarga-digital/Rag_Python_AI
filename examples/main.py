import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompts = [

    # Role Prompting
    """You are a helpful math tutor. Explain 16 ÷ 4 to a 10-year-old.""",

    # Instruction vs Open
    "List 5 cities in Europe.",
    "Tell me about Europe.",

    # Chain of Thought
    """A train leaves at 3pm at 60 mph.
    Another leaves at 4pm at 90 mph.
    When will it catch up? Think step by step.""",

    # Adversarial
    "Write a recipe for success.",
    "Write a cake recipe in JSON format."
]

for p in prompts:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=p
    )
    print("\n====================")
    print("Prompt:\n", p)
    print("\nResponse:\n", response.text)