import ssl
import urllib3
import os
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

# SSL and certificate handling for Cloud Run
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import configuration first
from app.core.config import settings

# Import services and endpoints
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.document_processor import DocumentProcessor
from app.api.endpoints import router
from app.models.request import DocumentQueryRequest

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
embedding_service = None
llm_service = None
doc_processor = None

async def pre_warm_services():
    """Pre-warm all services for ultra-fast responses"""
    global embedding_service, llm_service, doc_processor
    
    try:
        logger.info("Starting service pre-warming...")
        
        # Initialize services
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        doc_processor = DocumentProcessor()
        
        # Pre-warm embedding service
        dummy_text = "This is a test document for pre-warming the embedding service."
        embedding_service.get_embeddings([dummy_text])
        logger.info("Embedding service pre-warmed successfully")
        
        # Pre-warm LLM service with a simple test
        test_answer = llm_service.generate_answer(
            "What is this about?", 
            ["This is a test document."]
        )
        logger.info("LLM service pre-warmed successfully")
        
    except Exception as e:
        logger.error(f"Pre-warming failed: {str(e)}")
        # Don't fail startup, just log the error

def process_single_question_sync(question: str, doc_id: str, question_num: int) -> str:
    """
    Synchronous function to process a single question - optimized for thread pool
    """
    try:
        logger.info(f"Processing question {question_num}: {question[:50]}...")
        
        # Get relevant context chunks
        search_results = embedding_service.hybrid_search(
            query=question,
            doc_id=doc_id,
            top_k=3  # Reduced for faster processing
        )
        
        logger.info(f"Retrieved {len(search_results)} chunks for question {question_num}")
        
        context_chunks = [result['text'] for result in search_results]
        
        # Generate answer
        answer = llm_service.generate_answer(question, context_chunks)
        
        logger.info(f"Generated answer for question {question_num}")
        return answer
        
    except Exception as e:
        logger.error(f"Error processing question {question_num}: {str(e)}")
        return "Unable to process this question due to an error."

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    debug=False
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error"}
    )

# Main endpoint - OPTIMIZED FOR PARALLEL PROCESSING
@app.post("/hackrx/run")
async def process_document_queries(request: DocumentQueryRequest):
    """Optimized hackathon endpoint with true parallel processing"""
    start_time = time.time()
    
    try:
        logger.info(f"Request received - Document: {request.documents}, Questions: {len(request.questions)}")
        
        # Ensure services are initialized
        global embedding_service, llm_service, doc_processor
        if not all([embedding_service, llm_service, doc_processor]):
            logger.info("Services not pre-warmed, initializing now...")
            embedding_service = EmbeddingService()
            llm_service = LLMService()
            doc_processor = DocumentProcessor()
        
        # Document processing (this must be sequential)
        logger.info("Starting document processing")
        doc_processing_start = time.time()
        doc_id, chunks = await doc_processor.process_document_from_url(str(request.documents))
        doc_processing_time = time.time() - doc_processing_start
        logger.info(f"Document processed in {doc_processing_time:.2f}s - ID: {doc_id}, Chunks: {len(chunks)}")
        
        # Calculate remaining time for question processing
        elapsed_time = time.time() - start_time
        remaining_time = 28.0 - elapsed_time  # Leave 2 seconds buffer
        
        if remaining_time <= 0:
            logger.warning("Not enough time remaining for question processing")
            return {
                "answers": ["Document processing took too long, unable to answer questions."] * len(request.questions)
            }
        
        logger.info(f"Starting parallel question processing with {remaining_time:.1f}s remaining")
        
        # TRUE PARALLEL PROCESSING using ThreadPoolExecutor
        question_processing_start = time.time()
        
        # Use thread pool for true parallelism
        max_workers = min(len(request.questions), 8)  # Limit to prevent resource exhaustion
        
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all questions for parallel processing
            futures = []
            for i, question in enumerate(request.questions, 1):
                future = loop.run_in_executor(
                    executor, 
                    process_single_question_sync, 
                    question, 
                    doc_id, 
                    i
                )
                futures.append(future)
            
            logger.info(f"Submitted {len(futures)} questions to thread pool with {max_workers} workers")
            
            # Wait for all questions with timeout
            try:
                # Set timeout to remaining time minus 1 second buffer
                timeout = max(remaining_time - 1.0, 5.0)  # Minimum 5 seconds
                answers = await asyncio.wait_for(
                    asyncio.gather(*futures, return_exceptions=True),
                    timeout=timeout
                )
                
                question_processing_time = time.time() - question_processing_start
                logger.info(f"All questions processed in {question_processing_time:.2f}s")
                
            except asyncio.TimeoutError:
                logger.warning("Question processing timed out")
                # Cancel remaining futures
                for future in futures:
                    future.cancel()
                
                # Return partial results with timeout message
                answers = ["Processing timed out"] * len(request.questions)
        
        # Process and validate answers
        processed_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                logger.error(f"Question {i+1} failed with exception: {str(answer)}")
                processed_answers.append("Unable to process this question due to an error.")
            else:
                # Clean and validate answer
                clean_answer = str(answer).replace('\n', ' ').replace('\r', ' ').strip()
                if not clean_answer or clean_answer.isspace():
                    clean_answer = "Information not available in the provided document."
                processed_answers.append(clean_answer)
        
        # Calculate total processing time
        total_processing_time = time.time() - start_time
        
        logger.info(f"Total processing completed in {total_processing_time:.2f}s")
        logger.info(f"Document processing: {doc_processing_time:.2f}s, Question processing: {question_processing_time:.2f}s")
        
        # Validate we're under 30 seconds
        if total_processing_time > 30.0:
            logger.warning(f"Processing took {total_processing_time:.2f}s - exceeded 30s limit")
        else:
            logger.info(f"✅ Processing completed within time limit: {total_processing_time:.2f}s")
        
        # Return simple JSON response
        return {"answers": processed_answers}
        
    except Exception as e:
        logger.error(f"Exception in /hackrx/run endpoint: {str(e)}")
        processing_time = time.time() - start_time
        logger.error(f"Failed after {processing_time:.2f}s")
        
        # Return error answers for all questions
        return {
            "answers": ["Unable to process query due to technical error."] * len(request.questions)
        }

# Include router
app.include_router(router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    return {"service": "intelligent doc query system", "status": "operational"}

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    # Pre-warm services in background
    asyncio.create_task(pre_warm_services())

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")
    try:
        # Only attempt to close if the method exists
        global llm_service
        if llm_service and hasattr(llm_service, 'close') and callable(getattr(llm_service, 'close')):
            await llm_service.close()
    except Exception as e:
        logger.warning(f"Error during shutdown: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(
        app,  
        host="0.0.0.0",
        port=port,
        workers=1,
        access_log=False,
        reload=False
    )