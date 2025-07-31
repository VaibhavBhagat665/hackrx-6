import os
from urllib.parse import urlparse
from pydantic import BaseModel, HttpUrl, validator
from typing import List

class DocumentQueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError('questions cannot be empty')
        if len(v) > 20:
            raise ValueError('max 20 questions allowed')
        return v
    
    @validator('documents')
    def validate_document_url(cls, v):
        allowed_extensions = ['.pdf', '.docx']
        # Convert the HttpUrl object to a string before parsing
        path = urlparse(str(v)).path
        _, file_extension = os.path.splitext(path)
        
        if file_extension.lower() not in allowed_extensions:
            raise ValueError('only pdf and docx files supported')
        return v

class WebhookRequest(BaseModel):
    event_type: str
    payload: dict
    timestamp: str