Module 3: Chains and LCEL (LangChain Expression Language)

This script demonstrates how to chain components together using the pipe `|` operator.
This is the core of modern LangChain.

Concepts covered:
- LCEL: The `|` syntax.
- Runnables: Almost everything in LangChain is a Runnable.
- OutputParsers: Converting LLM output (Message) to a string or structured data.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return

    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    print("--- 1. Define Components ---")
    
    # Component 1: Prompt
    prompt = ChatPromptTemplate.from_template("Tell me a fun fact about {animal}.")
    
    # Component 2: Model
    # chat = ChatOpenAI(...) defined above
    
    # Component 3: Output Parser
    # The raw output of a ChatModel is a generic 'AIMessage'.
    # StrOutputParser extracts just the content string.
    parser = StrOutputParser()

    print("--- 2. Create Chain using LCEL ---")
    # The pipe operator '|' feeds the output of one component into the input of the next.
    # Flow: Input Dictionary -> Prompt -> ChatModel -> StrOutputParser -> String Output
    chain = prompt | chat | parser

    print("--- 3. Invoke Chain ---")
    # We invoke the chain just like we invoke the model, but the input is now verified against the prompt's variables.
    response = chain.invoke({"animal": "axolotls"})
    
    print("Response:")
    print(response)

    print("\n--- End of Module 3 ---")

if __name__ == "__main__":
    main()
