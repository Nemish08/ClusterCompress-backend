from pydantic import BaseModel
from typing import List, Optional

class CompressionAlgoResult(BaseModel):
    image_b64: str
    size: int
    palette: List[str]
    num_colors: int

class MultiCompressionResult(BaseModel):
    original_size: int
    kmeans: CompressionAlgoResult
    hierarchical: CompressionAlgoResult
    dbscan: CompressionAlgoResult