Module 5: Agents

This script demonstrates how to create a simple Agent.
Agents use an LLM to decide what actions to take and in what order. An action can be using a tool and observing its output.

Concepts covered:
- Tools (custom and built-in)
- create_tool_calling_agent
- AgentExecutor

"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return

    print("--- 1. Define Tools ---")
    
    # We can define custom tools using the @tool decorator
    @tool
    def get_word_length(word: str) -> int:
        """Returns the length of a word."""
        return len(word)
    
    @tool
    def add_numbers(a: int, b: int) -> int:
        """Adds two numbers together."""
        return a + b

    tools = [get_word_length, add_numbers]

    print("--- 2. Initialize Model ---")
    # We must use a model that supports tool calling (function calling)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    print("--- 3. Create Agent ---")
    # The prompt typically needs a placeholder for the agent scratchpad (intermediate steps)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You can use tools to help answer questions."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Construct the tool calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the executor (the runtime for the agent)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("--- 4. Run Agent ---")
    query = "What is the length of the word 'LangChain' plus the length of the word 'Python'?"
    print(f"Query: {query}")
    
    response = agent_executor.invoke({"input": query})
    
    print("\nFinal Answer:")
    print(response["output"])

    print("\n--- End of Module 5 ---")

if __name__ == "__main__":
    main()
