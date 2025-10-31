# AI Document Processing System

A local AI-powered system for intelligent PDF document processing, featuring automatic classification, structured data extraction, and semantic search capabilities.

## Features

- **Document Classification**: Automatically classifies PDFs into categories (Invoice, Resume, Utility Bill, Other)
- **Data Extraction**: Extracts structured data based on document type:
  - **Invoices**: Invoice number, date, company name, total amount
  - **Resumes**: Name, email, phone, years of experience
  - **Utility Bills**: Account number, date, usage (kWh), amount due
- **Semantic Search**: Vector-based semantic search using FAISS for document retrieval
- **REST API**: FastAPI-based endpoints for document upload and search
- **Local Processing**: All AI models run locally (no external API calls)

## Tech Stack

- **Python** 3.12+
- **FastAPI** - Web framework for REST API
- **LangChain** - Document processing and vector stores
- **Transformers** - Zero-shot classification
- **Sentence Transformers** - Text embeddings
- **FAISS** - Vector similarity search
- **PyMuPDF** - PDF text extraction
- **Pydantic** - Data validation
- **spaCy** - NLP processing

## Prerequisites

- Python 3.12 or higher
- `uv` package manager

To install `uv`:
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

### 1. Clone the repository (if applicable)
```bash
git clone <repository-url>
cd log
```

### 2. Install dependencies

```bash
uv sync
```

The project uses the following main dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `langchain-community` - Document processing
- `transformers==4.36.2` - NLP models
- `sentence-transformers==2.3.1` - Embeddings
- `torch>=2.9.0` - Deep learning framework
- `pdfplumber` - PDF parsing
- `pymupdf` - PDF processing
- `spacy==3.7.2` - NLP toolkit
- `pydantic` - Data validation
- `python-dotenv` - Environment management

### 3. (Optional) Download spaCy language model
```bash
python -m spacy download en_core_web_sm
```

## Running the Application

### Start the server

```bash
uv run python src/main.py
```

The API server will start on **http://localhost:8000**

### Access the API

- **Interactive API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## API Endpoints

### 1. Upload and Process Document
**Endpoint**: `POST /upload`

Upload a PDF file for classification and data extraction.

**Parameters**:
- `file`: PDF file to upload
- `index_name`: (optional) Name of the search index to store document in (default: "default")

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/upload?index_name=my_docs" \
  -F "file=@invoice.pdf"
```

**Response**:
```json
{
  "filename": "invoice.pdf",
  "index_name": "my_docs",
  "class": "Invoice",
  "invoice_number": "INV-12345",
  "date": "2024-01-15",
  "company": "Acme Corp Inc.",
  "total_amount": 1250.50
}
```

### 2. Search Documents
**Endpoint**: `POST /search`

Perform semantic search across indexed documents.

**Request Body**:
```json
{
  "index_name": "my_docs",
  "query": "find invoices from Acme Corp",
  "top_k": 5
}
```

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "index_name": "my_docs",
    "query": "invoices from January",
    "top_k": 5
  }'
```

**Response**:
```json
{
  "query": "invoices from January",
  "results": [
    {
      "rank": 1,
      "filename": "invoice.pdf",
      "similarity_score": 0.8542,
      "text_snippet": "Invoice #INV-12345 Date: 2024-01-15..."
    }
  ],
  "total_results": 1
}
```

## Libraries and Methods Used

### Document Processing
- **PyMuPDFLoader (LangChain)**: Extracts text from PDF documents using PyMuPDF backend
- **Regex-based extraction**: Pattern matching for structured data (dates, amounts, emails, phone numbers)
- **Text cleaning**: Normalization and whitespace removal

### Classification
- **Zero-shot classification**: Uses `facebook/bart-large-mnli` model from Hugging Face
- **Hybrid approach**: Combines ML predictions with rule-based keyword matching
- **Categories**: Invoice, Resume, Utility Bill, Other, Unclassifiable

### Semantic Search
- **FAISS (Facebook AI Similarity Search)**: Vector database for similarity search
- **Sentence Transformers**: `all-MiniLM-L6-v2` model for generating text embeddings
- **Cosine similarity**: Measures semantic similarity between query and documents

### Data Validation
- **Pydantic**: Schema validation for API requests and responses
- **Type safety**: Ensures data integrity throughout the pipeline

## Output Format

Processed documents are saved to `output.json` with the following structure:

```json
{
  "invoice.pdf": {
    "filename": "invoice.pdf",
    "index_name": "default",
    "class": "Invoice",
    "invoice_number": "INV-12345",
    "date": "2024-01-15",
    "company": "Acme Corp Inc.",
    "total_amount": 1250.50
  },
  "resume.pdf": {
    "filename": "resume.pdf",
    "index_name": "default",
    "class": "Resume",
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "555-123-4567",
    "experience_years": 5
  }
}
```

## How It Works

1. **Upload**: PDF is uploaded via `/upload` endpoint
2. **Text Extraction**: PyMuPDF extracts text from the PDF
3. **Classification**: Zero-shot classifier determines document type
4. **Data Extraction**: Regex patterns extract structured fields based on document type
5. **Indexing**: Document text is embedded and stored in FAISS vector store
6. **Search**: Semantic search converts queries to embeddings and finds similar documents
7. **Results**: Extracted data is returned via API and saved to `output.json`

## Notes

- First run will download ML models (~500MB-1GB total)
- Models run on CPU by default (GPU support available with CUDA-enabled PyTorch)
- FAISS indexes are saved to `data/models/` for persistence
- All processing happens locally - no external API calls

## Troubleshooting

**Models not downloading**: Ensure you have internet connection on first run
**Out of memory**: Reduce batch size or use smaller models
**CUDA errors**: Set device to CPU in classifier and retrieval modules
