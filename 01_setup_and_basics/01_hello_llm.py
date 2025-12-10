Module 1: Setup and Basics

This script demonstrates the absolute basics of using LangChain:
1. Loading environment variables (for API keys).
2. Initializing a Chat Model.
3. Sending a simple message to the model and printing the response.

Concepts covered:
- ChatModels: The standard interface for LLMs in LangChain.
- SystemMessage & HumanMessage: The basic message types.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Load Environment Variables
# This looks for a .env file in the project root and loads variables like OPENAI_API_KEY
load_dotenv()

def main():
    # Check if API Key is present
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
        return

    print("--- 1. Initializing Chat Model ---")
    # Initialize the ChatOpenAI model. 
    # 'model' specifies which OpenAI model to use (e.g., gpt-3.5-turbo, gpt-4)
    # 'temperature' controls randomness (0 is deterministic, 1 is creative)
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    print("--- 2. Creating Messages ---")
    # Messages are the input to Chat Models.
    # SystemMessage: Sets the behavior of the AI.
    # HumanMessage: The user's input.
    messages = [
        SystemMessage(content="You are a helpful assistant that speaks like a pirate."),
        HumanMessage(content="Hello! Tell me a short story about the sea."),
    ]

    print("--- 3. Invoking the Model ---")
    # We 'invoke' the model with our list of messages.
    response = chat.invoke(messages)

    print("\nResponse from AI:")
    print(response.content)

    print("\n--- End of Module 1 ---")

if __name__ == "__main__":
    main()
