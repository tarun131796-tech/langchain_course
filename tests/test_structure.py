import pytest
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to check for syntax errors and import errors
def test_imports():
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import chromadb

def test_module_scripts_exist():
    assert os.path.exists("01_setup_and_basics/01_hello_llm.py")
    assert os.path.exists("02_prompts_and_models/01_prompts.py")
    assert os.path.exists("03_chains_and_lcel/01_simple_chain.py")
    assert os.path.exists("04_rag_basics/01_rag.py")
    assert os.path.exists("05_agents/01_simple_agent.py")
