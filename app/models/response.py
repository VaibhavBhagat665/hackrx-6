from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryResponse(BaseModel):
    answers: List[str]
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class WebhookResponse(BaseModel):
    status: str
    message: str
    request_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime: Optional[float] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int
