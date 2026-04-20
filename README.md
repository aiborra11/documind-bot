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
You can execute the code by using either Docker or run it locally. 

### Pre-requisite: Launch Local LLM
Regardless of the deployment method, ensure Ollama is running natively on your host machine to leverage hardware acceleration (GPU), and pull the required model:
```bash
ollama serve
ollama pull phi3.5  # or your preferred model
```

### Environment Setup
```bash
# Clone and enter the directory
git clone <repository-url>
cd documind-bot
```

### Option A: Docker Deployment
```bash
# Build and start the container
docker-compose up --build -d
```

Verify the API is running:
Open your browser and navigate to http://localhost:8000/docs to see the interactive Swagger UI.


### Option B: Local Deployment
```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

```bash
# Start the API Using the provided script
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
### Step 4: Evaluate the System
You can test the accuracy of the RAG pipeline using the built-in evaluation endpoints.

**Option A: Full Batch Evaluation**
Runs the entire dataset located at `data/input_data/eval.jsonl` and returns overall KPIs (Hit Rate, Citation Accuracy, Response Overlap). *Note: This is a synchronous process and may take a few minutes depending on hardware.*

* **Request:** `POST /qa/evaluate/batch`
* **Body:** None
```json
{
  "status": "success",
  "summary": {
    "total_questions_analyzed": 2,
    "retrieval_hit_rate_percentage": 100,
    "valid_citation_rate_percentage": 50,
    "average_response_accuracy_percentage": 81.67,
    "average_latency_seconds": 79.18
  },
  "detailed_results": [
    {
      "query": "Is AMD growing the operating margin within the data center vertical from 2024 to 2025?",
      "metrics": {
        "retrieval_hit": true,
        "final_answer": "- Yes, according to Document [1], on Page: 21, there is an increase in both revenue and Operating Margin for Q4 2025 compared to Q4 2024. The operating margin increased from $3.9B (assumed as a percentage of the lower figure) with no specific value given but implied growth due to context, up to an actual dollar amount increase in Operating Income which is not directly stated here; however, it can be inferred that there was positive financial performance leading into Q4 2025.\n- Source: [1] AMD Q4'25 Earnings Slides FINAL.pdf | Page: 21",
        "expected_answer": "Yes, AMD is grew a 3% from Q4 2024 to Q4 2025",
        "matching_kw": [
          "3",
          "from",
          "2024",
          "2025"
        ],
        "valid_citation": true,
        "overlap_score": 0.8,
        "latency_seconds": 88.38
      }
    },
    {
      "query": "What is the EBITDA CAGR growth for Zedcor?",
      "metrics": {
        "retrieval_hit": true,
        "final_answer": "According to Document [2], which states, \"Adj. EBITDA (Ms) - Revenue and Adj. EBITDA excludes discontinued energy services operations sold in 2021 for $11.3M,\" the company's adjusted earnings before interest, taxes, depreciation, and amortization (Adjusted EBITDA), which is used as a non-IFRS standard measure to illustrate financial performance by management, has grown at an annual compound growth rate (CAGR) of 44% over the specified period.\n\nSource: [Document 2], Page: 3",
        "expected_answer": "Zedcor has grown at a rate of 44% Adj. EBITDA CAGR",
        "matching_kw": [
          "cagr",
          "grown",
          "rate",
          "44",
          "ebitda"
        ],
        "valid_citation": false,
        "overlap_score": 0.83,
        "latency_seconds": 69.99
      }
    }
  ]
}
```

**Option B: Single Query Evaluation**
Test a single question against an expected answer to see detailed overlap metrics.

* **Request:** `POST /qa/evaluate`
* **Payload (JSON):**
```json
{
  "query": "What is the EBITDA CAGR growth for Zedcor?",
  "expected_answer": "Zedcor has grown at a rate of 44% Adj. EBITDA CAGR",
  "expected_source": "2025-12-15-Investor-Slide-Deck.pdf",
  "expected_page": "3",
  "initial_top_k": 10,
  "final_top_n": 3
}
```


