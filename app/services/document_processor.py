import hashlib
import requests
import asyncio
import concurrent.futures
from typing import Tuple, Dict, List, Optional
from app.utils.document_parser import DocumentParser
from app.utils.text_processor import TextProcessor
from app.services.embedding_service import EmbeddingService
from app.core.logging import logger
from app.core.config import settings
import os
import time
import re
from urllib.parse import urlparse

class DocumentProcessor:
    """Enhanced document processing with parallel processing and better accuracy"""
    
    def __init__(self):
        self.parser = DocumentParser()
        self.text_processor = TextProcessor()
        self.embedding_service = EmbeddingService()
        self.processing_stats = {}
        
    async def process_document_from_url(self, url: str) -> Tuple[str, List[Dict]]:
        """Enhanced document processing with parallel processing and better error handling"""
        
        start_time = time.time()
        logger.info(f"Starting enhanced document processing for: {url}")
        
        try:
            # Download document with retries
            content, file_extension = await self._download_document_with_retry(url)
            
            # Parse document with enhanced error handling - FIXED: Removed await for sync methods
            text, metadata = await self._parse_document_enhanced(content, file_extension)
            
            # Create enhanced document ID
            doc_id = self._generate_enhanced_doc_id(url, metadata)
            
            # Enhanced parallel text processing
            chunks = await self._process_text_parallel(text, metadata)
            
            # Add enhanced metadata to chunks
            enhanced_chunks = self._enhance_chunks_metadata(chunks, doc_id, url, metadata)
            
            # Parallel embedding generation and storage
            success = await self._store_embeddings_parallel(doc_id, enhanced_chunks)
            
            if not success:
                raise Exception("Failed to store document embeddings")
            
            processing_time = time.time() - start_time
            
            # Store processing statistics
            self.processing_stats[doc_id] = {
                'processing_time': processing_time,
                'chunks_count': len(enhanced_chunks),
                'document_size': len(text),
                'pages': metadata.get('pages', 1),
                'accuracy_score': self._calculate_accuracy_score(metadata, enhanced_chunks)
            }
            
            logger.info(f"Enhanced processing completed for {doc_id}: {len(enhanced_chunks)} chunks in {processing_time:.2f}s")
            
            return doc_id, enhanced_chunks
            
        except Exception as e:
            logger.error(f"Enhanced document processing failed: {str(e)}")
            raise
    
    async def _download_document_with_retry(self, url: str, max_retries: int = 3) -> Tuple[bytes, str]:
        """Download document with retry logic and better error handling"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading document (attempt {attempt + 1}): {url}")
                
                # Use asyncio to run the blocking request in a thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        url, 
                        timeout=60,
                        headers={
                            'User-Agent': 'DocumentQuerySystem/2.0 (Enhanced)',
                            'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain'
                        }
                    )
                )
                response.raise_for_status()
                
                # Determine file extension
                path = urlparse(url).path
                _, file_extension = os.path.splitext(path)
                
                # Validate content type
                content_type = response.headers.get('content-type', '').lower()
                if not file_extension:
                    if 'pdf' in content_type:
                        file_extension = '.pdf'
                    elif 'word' in content_type or 'officedocument' in content_type:
                        file_extension = '.docx'
                    else:
                        file_extension = '.txt'
                
                logger.info(f"Document downloaded successfully: {len(response.content)} bytes, type: {file_extension}")
                return response.content, file_extension
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _parse_document_enhanced(self, content: bytes, file_extension: str) -> Tuple[str, Dict]:
        """Enhanced document parsing with proper async handling"""
        try:
            logger.info(f"Parsing document with enhanced parser: {file_extension}")
            
            # Run parsing in thread pool since parser methods are synchronous
            loop = asyncio.get_event_loop()
            
            if file_extension.lower() == '.pdf':
                # Check if parse_pdf_enhanced exists
                if hasattr(self.parser, 'parse_pdf_enhanced'):
                    method = getattr(self.parser, 'parse_pdf_enhanced')
                    # All parser methods are synchronous, run in thread pool
                    text, metadata = await loop.run_in_executor(None, method, content)
                else:
                    # Fallback to standard parse_pdf method
                    parse_method = getattr(self.parser, 'parse_pdf', lambda x: (x.decode('utf-8', errors='ignore'), {}))
                    text, metadata = await loop.run_in_executor(None, parse_method, content)
                    
            elif file_extension.lower() == '.docx':
                # Check if parse_docx_enhanced exists
                if hasattr(self.parser, 'parse_docx_enhanced'):
                    method = getattr(self.parser, 'parse_docx_enhanced')
                    # All parser methods are synchronous, run in thread pool
                    text, metadata = await loop.run_in_executor(None, method, content)
                else:
                    # Fallback to standard parse_docx method
                    parse_method = getattr(self.parser, 'parse_docx', lambda x: (x.decode('utf-8', errors='ignore'), {}))
                    text, metadata = await loop.run_in_executor(None, parse_method, content)
                    
            elif file_extension.lower() == '.txt':
                text = content.decode('utf-8', errors='ignore')
                metadata = {
                    'doc_type': 'txt',
                    'word_count': len(text.split()),
                    'char_count': len(text),
                    'pages': 1
                }
            else:
                raise ValueError(f"Unsupported document format: {file_extension}")
            
            # Ensure metadata has required fields
            if not isinstance(metadata, dict):
                metadata = {}
            
            metadata.setdefault('doc_type', file_extension.lower().lstrip('.'))
            metadata.setdefault('word_count', len(text.split()) if text else 0)
            metadata.setdefault('char_count', len(text) if text else 0)
            metadata.setdefault('pages', 1)
            
            logger.info(f"Document parsed successfully: {len(text)} characters")
            return text, metadata
            
        except Exception as e:
            logger.error(f"Document parsing failed: {str(e)}")
            raise
    
    async def _process_text_parallel(self, text: str, metadata: Dict) -> List[Dict]:
        """Process text into chunks using parallel processing for speed"""
        try:
            logger.info("Starting enhanced parallel text processing")
            
            # Use thread pool for CPU-intensive text processing
            loop = asyncio.get_event_loop()
            
            # Check if smart_chunk_enhanced method exists
            if hasattr(self.text_processor, 'smart_chunk_enhanced'):
                chunk_method = self.text_processor.smart_chunk_enhanced
            else:
                # Fallback to basic chunking if enhanced method doesn't exist
                chunk_method = getattr(self.text_processor, 'chunk_text', self._basic_chunk_text)
            
            chunks = await loop.run_in_executor(None, chunk_method, text, metadata)
            
            logger.info(f"Text processing completed: {len(chunks)} chunks generated")
            return chunks
            
        except Exception as e:
            logger.error(f"Parallel text processing failed: {str(e)}")
            # Fallback to basic chunking
            return self._basic_chunk_text(text, metadata)
    
    def _basic_chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Basic text chunking fallback method"""
        if not text:
            return []
        
        # Simple chunking: split into paragraphs or fixed-size chunks
        paragraphs = text.split('\n\n')
        chunks = []
        
        current_chunk = ""
        chunk_size = 1000  # characters
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_idx': 0,  # Simplified for basic implementation
                    'end_idx': len(current_chunk),
                    'chunk_type': 'paragraph_group'
                })
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'start_idx': 0,
                'end_idx': len(current_chunk),
                'chunk_type': 'paragraph_group'
            })
        
        return chunks
    
    def _enhance_chunks_metadata(self, chunks: List[Dict], doc_id: str, url: str, metadata: Dict) -> List[Dict]:
        """Add enhanced metadata to chunks for better search and retrieval"""
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            enhanced_chunk = {
                **chunk,
                'doc_id': doc_id,
                'source_url': url,
                'doc_metadata': metadata,
                'chunk_quality_score': self._calculate_chunk_quality(chunk),
                'processing_timestamp': time.time(),
                'enhanced_features': {
                    'has_numbers': bool(re.search(r'\d+', chunk.get('text', ''))),
                    'has_dates': bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', chunk.get('text', ''))),
                    'sentence_count': len(re.split(r'[.!?]+', chunk.get('text', ''))),
                    'avg_sentence_length': len(chunk.get('text', '').split()) / max(1, len(re.split(r'[.!?]+', chunk.get('text', ''))))
                }
            }
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    async def _store_embeddings_parallel(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Store embeddings using parallel processing for faster storage"""
        try:
            logger.info(f"Starting parallel embedding storage for {doc_id}")
            
            # Use thread pool for I/O intensive embedding operations
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, 
                self.embedding_service.store_document_vectors, 
                doc_id, 
                chunks
            )
            
            logger.info(f"Parallel embedding storage completed for {doc_id}: {'Success' if success else 'Failed'}")
            return success
            
        except Exception as e:
            logger.error(f"Parallel embedding storage failed: {str(e)}")
            # Fallback to synchronous storage
            try:
                return self.embedding_service.store_document_vectors(doc_id, chunks)
            except Exception as fallback_error:
                logger.error(f"Fallback embedding storage also failed: {str(fallback_error)}")
                return False
    
    def _generate_enhanced_doc_id(self, url: str, metadata: Dict) -> str:
        """Generate enhanced document ID with metadata consideration"""
        # Include metadata in ID generation for better uniqueness
        id_content = f"{url}_{metadata.get('pages', 0)}_{metadata.get('word_count', 0)}"
        return hashlib.md5(id_content.encode()).hexdigest()[:16]
    
    def _calculate_chunk_quality(self, chunk: Dict) -> float:
        """Calculate quality score for chunk to prioritize better content"""
        text = chunk.get('text', '')
        
        # Base quality metrics
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        
        quality_score = 0.5  # Base score
        
        # Word count scoring (optimal range 50-300 words)
        if 50 <= word_count <= 300:
            quality_score += 0.2
        elif word_count > 300:
            quality_score += 0.1
        
        # Sentence structure scoring
        if sentence_count >= 2:
            quality_score += 0.2
        
        # Content richness scoring
        if re.search(r'\d+', text):  # Contains numbers
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _calculate_accuracy_score(self, metadata: Dict, chunks: List[Dict]) -> float:
        """Calculate overall accuracy score for document processing"""
        try:
            # Base score from successful processing
            accuracy = 0.7
            
            # Bonus for complete processing
            if chunks and len(chunks) > 0:
                accuracy += 0.1
            
            # Bonus for rich metadata
            if metadata.get('pages', 0) > 0:
                accuracy += 0.1
            
            # Bonus for quality chunks
            avg_chunk_quality = sum(chunk.get('chunk_quality_score', 0.5) for chunk in chunks) / max(1, len(chunks))
            accuracy += avg_chunk_quality * 0.1
            
            return min(1.0, accuracy)
            
        except Exception:
            return 0.5
    
    def get_processing_stats(self, doc_id: str) -> Optional[Dict]:
        """Get processing statistics for a document"""
        return self.processing_stats.get(doc_id)
    
    def cleanup_old_stats(self, max_age_hours: int = 24):
        """Clean up old processing statistics"""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        docs_to_remove = []
        for doc_id, stats in self.processing_stats.items():
            if stats.get('processing_timestamp', current_time) < cutoff_time:
                docs_to_remove.append(doc_id)
        
        for doc_id in docs_to_remove:
            del self.processing_stats[doc_id]
        
        if docs_to_remove:
            logger.info(f"Cleaned up statistics for {len(docs_to_remove)} old documents")