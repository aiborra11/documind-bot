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

class EvalRequest(BaseModel):
    """Schema for evaluating a single question-answer pair."""
    query: str
    expected_answer: str
    expected_source: str
    expected_page: str
    initial_top_k: int = Field(default=10, description="Documents fetched from Vector DB")
    final_top_n: int = Field(default=3, description="Documents kept after re-ranking")
    threshold: Optional[float] = Field(default=None, description="Minimum similarity score")