Module 2: Prompts and Models

This script explores:
1. PromptTemplates: Reusable templates for generating prompts.
2. ChatPromptTemplate: Templates specifically for chat models (System/Human/AI messages).
3. Substitution: How to inject variables into templates.

Concepts covered:
- PromptTemplate vs ChatPromptTemplate
- invoke() with dictionary input
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv()

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return

    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    print("--- 1. Simple PromptTemplate ---")
    # Useful for simple string prompts (often used with legacy LLMs, but still relevant)
    template = "Tell me a {adjective} joke about {topic}."
    prompt = PromptTemplate.from_template(template)
    
    # Format the prompt to see what it looks like
    formatted_prompt = prompt.format(adjective="funny", topic="chickens")
    print(f"Formatted String Prompt: {formatted_prompt}")


    print("\n--- 2. ChatPromptTemplate ---")
    # The preferred way for Chat Models. It structures the conversation.
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant specialized in {domain}."),
        ("human", "Explain {concept} in simple terms.")
    ])

    print("--- 3. Invoking Chain (Prompt -> Model) manually ---")
    # We can format the messages and then pass them to the model
    messages = chat_template.format_messages(domain="physics", concept="quantum entanglement")
    print(f"Formatted Messages: {messages}")
    
    response = chat.invoke(messages)
    print("\nResponse:")
    print(response.content)

    print("\n--- End of Module 2 ---")

if __name__ == "__main__":
    main()
