Module 4: RAG (Retrieval Augmented Generation)

This script demonstrates the RAG pipeline:
1. Load data.
2. Split data into chunks.
3. Embed chunks and store in a vector database (Chroma).
4. Retrieve relevant chunks based on a query.
5. Generate an answer using the retrieved context.

Concepts covered:
- Document Loaders
- Text Splitters
- Embeddings
- Vector Stores (Chroma)
- Retrievers
- RunnableParallel (for passing context and question simultaneously)
"""

import os
import shutil
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return

    # Clean up existing vector store if it exists (for demo purposes)
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    print("--- 1. Load Document ---")
    # We'll create a dummy file to load
    with open("sample_data.txt", "w") as f:
        f.write("""
        LangChain is a framework for developing applications powered by language models.
        It enables applications that:
        1. Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
        2. Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)
        The main value props of LangChain are:
        1. Components: abstraction for working with language models, along with a collection of implementations for each abstraction.
        2. Off-the-shelf chains: a structured assembly of components for accomplishing specific higher-level tasks.
        """)
    
    loader = TextLoader("sample_data.txt")
    docs = loader.load()

    print("--- 2. Split Document ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    print("--- 3. Embed and Store ---")
    # We use OpenAIEmbeddings to convert text to vectors
    embeddings = OpenAIEmbeddings()
    
    # Store in ChromaDB (local vector store)
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )

    print("--- 4. Retrieve ---")
    # Create a retriever interface
    retriever = vectorstore.as_retriever()
    
    # Define the RAG prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    print("--- 5. Build Chain ---")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RunnablePassthrough allows us to pass the user's question to both the retriever AND the prompt
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    print("--- 6. Invoke Chain ---")
    question = "What are the main value props of LangChain?"
    response = chain.invoke(question)
    
    print(f"Question: {question}")
    print(f"Answer: {response}")

    # Cleanup
    if os.path.exists("sample_data.txt"):
        os.remove("sample_data.txt")

    print("\n--- End of Module 4 ---")

if __name__ == "__main__":
    main()
