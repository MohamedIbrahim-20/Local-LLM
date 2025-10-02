# Local-LLM

# 🤖 RAG Comparison System

A modern Retrieval-Augmented Generation (RAG) system that compares **LangChain** and **LlamaIndex** implementations side-by-side. Perfect for understanding different RAG approaches and choosing the right framework for your project.

## ✨ Features

- **Three RAG Implementations:**
  - LangChain Manual Pipeline (full control)
  - LangChain RetrievalQA Chain (high-level abstraction)
  - LlamaIndex (optimized RAG framework)

- **CSV Document Processing:** Load and query structured data from CSV files
- **Vector Storage:** FAISS-based vector database with persistence
- **Powered by Groq:** Fast LLM inference with Llama 3.3 70B
- **Side-by-Side Comparison:** Run all three methods with one query

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MohamedIbrahim-20/rag-comparison.git
cd rag-comparison
```

2. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

3. Prepare your data:
```bash
# Place your CSV file in the project root
# Default: tech_100_long_real.csv
```

### Usage

Run the comparison system:

```bash
python rag.py
```

Enter your question when prompted, and the system will:
1. Load your documents
2. Create/load vector embeddings
3. Run all three RAG methods
4. Display answers from each approach

## 📁 Project Structure

```
.
├── rag.py                          # Main script
├── tech_100_long_real.csv         # Sample data (your CSV here)
├── .env                           # API keys
├── requirements.txt               # Dependencies
└── storage/
    ├── langchain_faiss/           # LangChain vector store
    └── llamaindex_storage/        # LlamaIndex vector store
```

## 🔧 Configuration

Edit the configuration section in `rag.py`:

```python
# Model Configuration
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 5
TEMPERATURE = 0.1

# CSV Configuration
CSV_FILE = "./tech_100_long_real.csv"
```

## 📊 CSV Format

Your CSV should contain at least:
- `title`: Document title
- `content`: Main content
- Additional metadata columns (optional)

Example:
```csv
title,content,category,date
"Python Best Practices","Write clean, maintainable code...",programming,2024-01-01
```

## 🛠️ Dependencies

```
langchain
langchain-community
langchain-groq
langchain-huggingface
llama-index
llama-index-llms-groq
llama-index-embeddings-huggingface
faiss-cpu
pandas
python-dotenv
```

## 🔑 API Keys

Get a free Groq API key:
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for an account
3. Generate an API key
4. Add to `.env`: `GROQ_API_KEY=your_key_here`

## 📝 Example Output

```
🤖 RAG COMPARISON SYSTEM
============================================================
Model: llama-3.3-70b-versatile
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Chunk Size: 1000 | Overlap: 100
Top-K Results: 5
============================================================

❓ Enter your question: What are the latest developments in AI?

============================================================
🔧 LangChain Manual RAG Pipeline
============================================================
📄 Split into 150 chunks
📂 Loading existing vector store
🔍 Retrieved 5 relevant chunks

💡 Answer:
[Answer from LangChain Manual method...]

============================================================
⚡ LangChain RetrievalQA Chain
============================================================
📄 Split into 150 chunks

💡 Answer:
[Answer from LangChain QA Chain...]

============================================================
🚀 LlamaIndex RAG
============================================================
📂 Loading existing index

💡 Answer:
[Answer from LlamaIndex...]

✅ All RAG methods completed successfully!
```


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
