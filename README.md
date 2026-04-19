# Documind Bot: Financial Analysis RAG System

**Documind Bot** is a Retrieval-Augmented Generation (RAG) pipeline designed to extract precise insights from financial statements and complex documents. It leverages a two-stage retrieval architecture to ensure maximum accuracy and verifiable citations.

## Tech Stack
- **Framework:** FastAPI (Asynchronous)
- **Orchestration:** LangChain (LCEL)
- **Vector Store:** ChromaDB
- **Retrieval:** Semantic Search + Cross-Encoder Re-ranking (`ms-marco-MiniLM-L-6-v2`)
- **LLM Provider:** Local Ollama integration
- **Ingestion:** Hybrid extraction (PyMuPDF + MarkItDown)

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/ingestion/embed-pdf` | `POST` | **Upload & Ingest**: Accepts a PDF, saves it, and starts the async embedding pipeline. Returns a `task_id`. |
| `/ingestion/status/{task_id}`| `GET` | **Task Tracking**: Checks if the background ingestion is `pending`, `processing`, or `completed`. |
| `/ingestion/tasks` | `GET` | **Monitoring**: Lists all recent ingestion tasks and their current status. |
| `/ingestion/reset-db` | `DELETE` | **Maintenance**: Completely wipes the vector database for a fresh start. |
| `/qa/ask` | `POST` | **Question & Answering**: Performs semantic search, re-ranks results, and generates an answer with citations. |

---

## Execution Workflow

### 1. Environment Setup
```bash
# Clone and enter the directory
git clone <repository-url>
cd documind-bot

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Local LLM
```bash
# Ensure Ollama is running and pull the required model:
ollama serve
ollama pull llama3  # or your preferred model
```

### 3. Start the API
```bash
# Using the provided script
chmod +x start_local.sh
./start_local.sh

# Or directly via uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```



## How to Use it

### Step 1: Ingest a Document
Upload your PDF. The process is asynchronous to handle large financial reports without timeouts.

* **Request:** `POST /ingestion/embed-pdf`
* **Body (form-data):** `file=@your_report.pdf`
* **Action:** Save the `task_id` returned in the response to track progress.

### Step 2: Monitor Status
Check if the document processing (extraction, chunking, and embedding) is finished before querying.

* **Request:** `GET /ingestion/status/<task_id>`
* **Expected Status:** `completed`

### Step 3: Perform Q&A
Ask a natural language question based on the ingested data. The system will retrieve context, re-rank it, and generate a cited answer.

* **Request:** `POST /qa/ask`
* **Payload (JSON):**
```json
{
  "query": "What were the net revenues for Q4?",
  "initial_top_k": 10,
  "final_top_n": 3
}
```
* **Expected Response:** The system returns a generated answer with explicit Source and Page references based on the context.

```json
{
  "status": "success",
  "query": "What happened with the FDA and hims and hers?",
  "answer": "- In September 2025, approximately 50 companies marketing compounded GLP-1 products received warning letters from the FDA [Source: hh10k.pdf, Page: 9]. The Company also received two such warnings directly related to its operations with Hims & Hers branded products containing similar active pharmaceutical ingredients as mentioned in Document [2], which aligns with this context of the FDA's actions towards companies involved in compounded drugs.\n- In January 2026, there was a temporary draw on $150 million from their Credit Facility to facilitate merger activities related to YourBio as stated in Document [3]. This information is not directly about the FDA's actions but provides context for financial maneuvers around that time.\n- In February 2026, following an announcement by HHS regarding potential violations of Federal Food, Drug, and Cosmetic Act (referenced in Document [3]), there was a referral to the DOJ which could potentially involve FDA regulations as well since it concerns pharmaceutical practices.\n- The Company has not received any warning letters from the FDA specifically related to Hims & Hers branded products following these events, according to information provided in Document [3]. However, there is an ongoing SEC Investigation that began around February 2026 (Document [3]) which may have implications for compliance with securities laws and could intersect regulatory concerns from the FDA.\n- The outcome of these events cannot be predicted at this time as per Documents [3] & [4].\n[Source: hh10k.pdf, Pages: 9; 131]",
  "sources_used": 3,
  "context_metadata": [
    {
      "page": 12,
      "source": "hh10k.pdf"
    },
    {
      "page": 12,
      "source": "hh10k.pdf"
    },
    {
      "page": 131,
      "source": "hh10k.pdf"
    }
  ]
}
```

