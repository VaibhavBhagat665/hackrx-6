import hashlib
import requests
from typing import Tuple, Dict, List
import asyncio
from app.utils.document_parser import DocumentParser
from app.utils.text_processor import TextProcessor
from app.services.embedding_service import EmbeddingService
from app.core.logging import logger
import os
from urllib.parse import urlparse

class DocumentProcessor:
    """main document processing orchestrator"""
    
    def __init__(self):
        self.parser = DocumentParser()
        self.text_processor = TextProcessor()
        self.embedding_service = EmbeddingService()
    
    async def process_document_from_url(self, url: str) -> Tuple[str, List[Dict]]:
        """download and process document from url"""
        
        try:
            # download document
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Use urlparse to get the path and then splitext to get the extension
            path = urlparse(url).path
            _, file_extension = os.path.splitext(path)
            
            # determine document type and parse
            if file_extension.lower() == '.pdf':
                text, metadata = self.parser.parse_pdf(response.content)
            elif file_extension.lower() == '.docx':
                text, metadata = self.parser.parse_docx(response.content)
            else:
                raise ValueError("unsupported document format")
            
            # create document id
            doc_id = hashlib.md5(url.encode()).hexdigest()[:12]
            
            # process text into chunks
            chunks = self.text_processor.smart_chunk(text)
            
            # add metadata to chunks
            for chunk in chunks:
                chunk.update({
                    'doc_id': doc_id,
                    'source_url': url,
                    'doc_metadata': metadata
                })
            
            # store embeddings
            success = self.embedding_service.store_document_vectors(doc_id, chunks)
            
            if not success:
                raise Exception("failed to store document vectors")
            
            logger.info(f"processed document {doc_id} with {len(chunks)} chunks")
            
            return doc_id, chunks
            
        except Exception as e:
            logger.error(f"document processing failed: {str(e)}")
            raise