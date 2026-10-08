import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )

print("Gemini API key loaded successfully!")

client = genai.Client(api_key=api_key)

print("Gemini client initialized!")

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Explain what cybersecurity means in simple words."
)

print("\nGemini Response:")
print(interaction.output_text)