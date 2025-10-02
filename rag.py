"""
Modern RAG (Retrieval-Augmented Generation) System
Compares LangChain and LlamaIndex implementations
"""

import os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document as LlamaDocument,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
UserWarning = False
# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-api-key-here")

# Model Configuration
LLM_MODEL = "llama-3.3-70b-versatile"  # More powerful model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Faster, still accurate

# RAG Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 5
TEMPERATURE = 0.1

# Storage paths
LANGCHAIN_DB_PATH = "./storage/langchain_faiss"
LLAMAINDEX_DB_PATH = "./storage/llamaindex_storage"
# CSV_FILE = "historical_100_long.csv"
CSV_FILE = "./tech_100_long_real.csv"


# ═══════════════════════════════════════════════════════════
# LLAMAINDEX SETUP
# ═══════════════════════════════════════════════════════════

def configure_llamaindex():
    """Configure global LlamaIndex settings"""
    Settings.llm = Groq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=TEMPERATURE)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

import os
import pandas as pd
from typing import Tuple, List
from langchain.schema import Document
from llama_index.core import Document as LlamaDocument

def load_csv_to_documents(
    csv_path: str, 
    content_columns: Tuple[str, ...] = ("title", "content")
) -> Tuple[List[Document], List[LlamaDocument]]:
    """
    Load CSV and create documents for LangChain and LlamaIndex

    Args:
        csv_path: Path to CSV file
        content_columns: Columns to combine into document content

    Returns:
        Tuple of (LangChain documents, LlamaIndex documents)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV, fill NaNs with empty string
    df = pd.read_csv(csv_path).fillna("")
    
    # Combine content columns into a single string
    content_series = df[list(content_columns)].agg(" | ".join, axis=1)
    
    # Metadata = all other columns
    metadata_cols = [col for col in df.columns if col not in content_columns]
    metadata_list = df[metadata_cols].to_dict(orient="records")

    # Cast numeric metadata back to int where possible
    for meta in metadata_list:
        for key, val in meta.items():
            if isinstance(val, float) and val.is_integer():
                meta[key] = int(val)

    # Create LangChain Documents
    langchain_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(content_series, metadata_list)
    ]
    
    # Create LlamaIndex Documents
    llamaindex_docs = [
        LlamaDocument(text=text, metadata=meta)
        for text, meta in zip(content_series, metadata_list)
    ]
    
    print(f"✓ Loaded {len(langchain_docs)} documents from {csv_path}")
    return langchain_docs, llamaindex_docs



# ═══════════════════════════════════════════════════════════
# LANGCHAIN RAG - METHOD 1: Manual Pipeline
# ═══════════════════════════════════════════════════════════

def rag_langchain_manual(documents: List[Document], question: str) -> str:
    """
    LangChain RAG with manual control over retrieval and generation
    
    Steps:
    1. Split documents into chunks
    2. Create embeddings and vector store
    3. Retrieve relevant chunks
    4. Generate answer using custom prompt
    """
    print("\n" + "="*60)
    print("🔧 LangChain Manual RAG Pipeline")
    print("="*60)
    
    # Step 1: Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📄 Split into {len(chunks)} chunks")
    
    # Step 2: Create or load vector database
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if os.path.exists(LANGCHAIN_DB_PATH):
        print(f"📂 Loading existing vector store from {LANGCHAIN_DB_PATH}")
        vectorstore = FAISS.load_local(
            LANGCHAIN_DB_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print(f"🔨 Creating new vector store at {LANGCHAIN_DB_PATH}")
        os.makedirs(os.path.dirname(LANGCHAIN_DB_PATH), exist_ok=True)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(LANGCHAIN_DB_PATH)
    
    # Step 3: Retrieve relevant documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
    relevant_docs = retriever.invoke(question)
    print(f"🔍 Retrieved {len(relevant_docs)} relevant chunks")
    
    # Step 4: Build context from retrieved documents
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    print(f"\n=== Retrieved Context ===\n{context if context.strip() else '⚠️ No context found'}\n")
    # Step 5: Create custom prompt
    prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the context provided below.

If the answer is not in the context, respond with: "I don't have enough information to answer that question."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    # Step 6: Generate answer
    llm = ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)
    response = llm.invoke(prompt)
    
    answer = response.content
    print(f"\n💡 Answer:\n{answer}\n")
    
    return answer


# ═══════════════════════════════════════════════════════════
# LANGCHAIN RAG - METHOD 2: RetrievalQA Chain
# ═══════════════════════════════════════════════════════════

def rag_langchain_qa_chain(documents: List[Document], question: str) -> str:
    """
    LangChain RAG using built-in RetrievalQA chain
    
    This is a higher-level abstraction that handles retrieval and generation
    automatically using the "stuff" strategy (puts all docs in one prompt)
    """
    print("\n" + "="*60)
    print("⚡ LangChain RetrievalQA Chain")
    print("="*60)
    
    # Split and create vector store (same as manual method)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📄 Split into {len(chunks)} chunks")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if os.path.exists(LANGCHAIN_DB_PATH):
        vectorstore = FAISS.load_local(
            LANGCHAIN_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        os.makedirs(os.path.dirname(LANGCHAIN_DB_PATH), exist_ok=True)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(LANGCHAIN_DB_PATH)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    
    # Create QA chain with custom prompt
    prompt_template = """Use the following context to answer the question. If you cannot find the answer in the context, say so clearly.

Context: {context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Puts all retrieved docs into one prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    
    print(f"\n💡 Answer:\n{answer}\n")
    return answer


# ═══════════════════════════════════════════════════════════
# LLAMAINDEX RAG
# ═══════════════════════════════════════════════════════════

def rag_llamaindex(question: str) -> str:
    """
    LlamaIndex RAG implementation
    
    LlamaIndex provides a more streamlined API specifically designed for RAG
    with automatic optimization and better handling of large contexts
    """
    print("\n" + "="*60)
    print("🚀 LlamaIndex RAG")
    print("="*60)
    
    configure_llamaindex()
    
    # Load or create index
    if os.path.exists(LLAMAINDEX_DB_PATH):
        print(f"📂 Loading existing index from {LLAMAINDEX_DB_PATH}")
        storage_context = StorageContext.from_defaults(persist_dir=LLAMAINDEX_DB_PATH)
        index = load_index_from_storage(storage_context)
    else:
        print(f"🔨 Creating new index at {LLAMAINDEX_DB_PATH}")
        _, llamaindex_docs = load_csv_to_documents(CSV_FILE)
        index = VectorStoreIndex.from_documents(llamaindex_docs)
        os.makedirs(LLAMAINDEX_DB_PATH, exist_ok=True)
        index.storage_context.persist(persist_dir=LLAMAINDEX_DB_PATH)
    
    # Create query engine with optimized settings
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K_RESULTS,
        response_mode="compact",  # Optimizes token usage
    )
    
    # Query
    response = query_engine.query(question)
    answer = str(response)
    
    print(f"\n💡 Answer:\n{answer}\n")
    return answer


# ═══════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════

def main():
    """Run all three RAG implementations and compare results"""
    
    print("\n" + "="*60)
    print("🤖 RAG COMPARISON SYSTEM")
    print("="*60)
    print(f"Model: {LLM_MODEL}")
    print(f"Embeddings: {EMBEDDING_MODEL}")
    print(f"Chunk Size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    print(f"Top-K Results: {TOP_K_RESULTS}")
    print("="*60)
    
    # Get user question
    question = input("\n❓ Enter your question: ").strip()
    
    if not question:
        print("⚠️  No question provided. Exiting.")
        return
    
    # Load documents
    try:
        langchain_docs, _ = load_csv_to_documents(CSV_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Run all three methods
    try:
        rag_langchain_manual(langchain_docs, question)
        rag_langchain_qa_chain(langchain_docs, question)
        rag_llamaindex(question)
        
        print("\n" + "="*60)
        print("✅ All RAG methods completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during RAG execution: {e}")
        raise


if __name__ == "__main__":
    main()