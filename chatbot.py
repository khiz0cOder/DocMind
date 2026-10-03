from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This list stores the entire conversation history
messages = []

print("DocMind Chatbot - Type 'quit' to exit")
print("-" * 40)

while True:
    # Get user input
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    # Send full conversation history to AI
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=messages
    )
    
    # Get AI response
    ai_response = response.choices[0].message.content
    
    # Add AI response to history
    messages.append({"role": "assistant", "content": ai_response})
    
    print(f"DocMind: {ai_response}")
    print()