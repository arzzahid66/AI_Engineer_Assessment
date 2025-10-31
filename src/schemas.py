from typing import List, Optional
from pydantic import BaseModel, Field


# Pydantic models for API
class SearchRequest(BaseModel):
    """Request model for semantic search."""
    index_name: str = Field(..., description="Index name to search in")
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)


class SearchResult(BaseModel):
    """Response model for search results."""
    rank: int
    filename: str
    similarity_score: float
    text_snippet: Optional[str] = None


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str
    results: List[SearchResult]
    total_results: int


class StatusResponse(BaseModel):
    """Response model for system status."""
    status: str
    models_loaded: bool
    index_built: bool
    documents_indexed: int
    total_chunks: int