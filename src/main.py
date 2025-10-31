import os
import json,uvicorn
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from document_processor import DocumentProcessor
from classifier import DocumentClassifier
from extractor import DataExtractor
from retrieval import SemanticRetrieval
from schemas import SearchRequest,SearchResponse,SearchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Document Processing System",
    description="Local AI system for document classification, extraction, and semantic search"
)

# Global instances (loaded on startup)
processor = None
classifier = None
extractor = None
retrieval = None

# Configuration
INPUT_FOLDER = "data/input"
OUTPUT_FILE = "output.json"


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global processor, classifier, extractor, retrieval

    logger.info("Starting up API server...")

    try:
        # Create necessary directories
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        os.makedirs("data/models", exist_ok=True)

        # Initialize components
        logger.info("Loading document processor...")
        processor = DocumentProcessor()

        logger.info("Loading classifier model...")
        classifier = DocumentClassifier()

        logger.info("Loading extractor...")
        extractor = DataExtractor()

        logger.info("Loading retrieval system...")
        retrieval = SemanticRetrieval()

        # Try to load existing index
        # retrieval.load_index()

        logger.info("All models loaded successfully")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise



@app.post("/upload", tags=["Processing"])
async def upload_document(
    file: UploadFile = File(...),
    index_name: str = "default"
):
    """
    Upload and process a PDF document.

    Args:
        file: PDF file to upload
        index_name: Name of the index to store document in

    Returns:
        Classification and extracted data
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )

        # Save uploaded file temporarily
        temp_path = Path(INPUT_FOLDER) / file.filename
        with open(temp_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Processing file: {file.filename} for index: {index_name}")

        # Extract text
        text = processor.extract_text(str(temp_path))

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract text from PDF"
            )

        # Classify document
        doc_type = classifier.classify(text, file.filename)

        # Extract structured data
        extracted_data = extractor.extract(text, doc_type)

        # Add to search index
        retrieval.add_document(index_name, file.filename, text)

        # Prepare result
        result = {
            "filename": file.filename,
            "index_name": index_name,
            "class": doc_type,
            **extracted_data
        }

        # Save to output.json
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = {}

        all_results[file.filename] = result

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully processed: {file.filename}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(request: SearchRequest):
    """
    Search documents in a specific index.

    Args:
        index_name: Name of the index to search
        query: Search query
        top_k: Number of results to return
    """
    try:
        results = retrieval.search(request.index_name, request.query, request.top_k)

        if not results:
            return SearchResponse(
                query=request.query,
                results=[],
                total_results=0
            )

        search_results = [
            SearchResult(
                rank=r['rank'],
                filename=r['filename'],
                similarity_score=r['score'],
                text_snippet=r.get('text_snippet')
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")



if __name__ == "__main__":

    uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )