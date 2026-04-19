# documind-bot
Extract insights and answers from local files using LLMs.
My personal objective with this project is to develop a bot that could help me into my financial analysis tasks. I do analyse financial statements from company's annual/quarterly reports so this could result on a pretty helpful project.
Nevertheless, the bot must be able to process and Q&A any type of document.

# Document QA RAG Pipeline

A Retrieval-Augmented Generation (RAG) system designed for document-based question answering. This project implements a full pipeline: PDF ingestion, embeddings, semantic search with re-ranking, and LLM-powered responses with verifiable citations.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### 2. Installation
```bash


pip install -r requirements.txt
chmod +x start_local.sh
ollama serve
ollama pull <model>
./start_local.sh