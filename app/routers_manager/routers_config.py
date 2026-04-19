from typing import Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Payload schema for two-stage semantic search requests."""
    query: str = Field(
        ..., 
        min_length=3, 
        description="The natural language question to search for."
    )
    initial_top_k: Optional[int] = Field(
        default=10, 
        ge=1, 
        le=50, 
        description="Number of results to retrieve from the vector database (Stage 1)."
    )
    final_top_n: Optional[int] = Field(
        default=3, 
        ge=1, 
        le=20, 
        description="Number of results to return after cross-encoder re-ranking (Stage 2)."
    )
    threshold: Optional[float] = Field(
        default=1.0, 
        description="Optional distance threshold for Stage 1 filtering. Leave null to avoid strict cutoffs."
    )


