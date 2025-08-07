from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
import time
import traceback
import asyncio
import concurrent.futures
import threading
from typing import Dict, Any, List
from app.models.request import DocumentQueryRequest, WebhookRequest
from app.models.response import QueryResponse, WebhookResponse, HealthResponse
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService  
from app.services.llm_service import LLMService
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()

# Initialize services as singletons for thread safety
_doc_processor = None
_embedding_service = None
_llm_service = None
_services_lock = threading.Lock()

def get_services():
    """Thread-safe service initialization"""
    global _doc_processor, _embedding_service, _llm_service
    
    with _services_lock:
        if _doc_processor is None:
            _doc_processor = DocumentProcessor()
            _embedding_service = EmbeddingService()
            _llm_service = LLMService()
            logger.info("Services initialized successfully")
    
    return _doc_processor, _embedding_service, _llm_service

def process_single_question_optimized(question: str, doc_id: str, question_num: int) -> str:
    """
    Ultra-optimized single question processing for maximum speed
    
    Args:
        question: The question to process
        doc_id: Document ID for context retrieval
        question_num: Question number for logging
        
    Returns:
        Formatted answer string
    """
    try:
        start_time = time.time()
        logger.info(f"Processing Q{question_num}: {question[:40]}...")
        
        # Get services (thread-safe)
        _, embedding_service, llm_service = get_services()
        
        # Get relevant context chunks with a higher top_k for better accuracy
        search_start = time.time()
        search_results = embedding_service.hybrid_search(
            query=question,
            doc_id=doc_id,
            top_k=settings.max_context_chunks  # Use configured value (now 5)
        )
        search_time = time.time() - search_start
        
        logger.info(f"Q{question_num}: Retrieved {len(search_results)} chunks in {search_time:.2f}s")
        
        context_chunks = [result['text'] for result in search_results]
        
        # Generate answer using optimized LLM service
        llm_start = time.time()
        answer = llm_service.generate_answer(question, context_chunks)
        llm_time = time.time() - llm_start
        
        total_time = time.time() - start_time
        logger.info(f"Q{question_num}: Completed in {total_time:.2f}s (search: {search_time:.2f}s, llm: {llm_time:.2f}s)")
        
        return answer
        
    except Exception as e:
        logger.error(f"Error processing Q{question_num}: {str(e)}")
        return "Unable to process this question due to an error."

@router.post("/hackrx/run")
async def process_document_queries_ultra_fast(request: DocumentQueryRequest):
    """Ultra-optimized hackathon endpoint for sub-30 second processing"""
    overall_start_time = time.time()
    
    try:
        logger.info(f"🚀 ULTRA-FAST REQUEST - Document: {request.documents}, Questions: {len(request.questions)}")
        
        # Get services (initialize if needed)
        doc_processor, embedding_service, llm_service = get_services()
        
        # Phase 1: Document processing (sequential - required)
        doc_start_time = time.time()
        logger.info("📄 Starting document processing...")
        
        doc_id, chunks = await doc_processor.process_document_from_url(str(request.documents))
        
        doc_processing_time = time.time() - doc_start_time
        logger.info(f"📄 Document processed in {doc_processing_time:.2f}s - ID: {doc_id}, Chunks: {len(chunks)}")
        
        # Check time remaining
        elapsed_time = time.time() - overall_start_time
        remaining_time = 28.0 - elapsed_time  # 2-second buffer
        
        if remaining_time <= 2.0:
            logger.warning(f"⚠️ Insufficient time remaining: {remaining_time:.1f}s")
            return {
                "answers": ["Document processing took too long"] * len(request.questions)
            }
        
        # Phase 2: Ultra-parallel question processing
        questions_start_time = time.time()
        logger.info(f"⚡ Starting ULTRA-PARALLEL processing of {len(request.questions)} questions with {remaining_time:.1f}s remaining")
        
        # Determine optimal number of workers
        max_workers = min(len(request.questions), settings.max_concurrent_requests)  # Use configured value
        
        # Use ThreadPoolExecutor for true parallelism
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            logger.info(f"🔥 Launching {max_workers} parallel workers")
            
            # Submit all questions immediately
            futures = []
            submission_start = time.time()
            
            for i, question in enumerate(request.questions, 1):
                future = loop.run_in_executor(
                    executor, 
                    process_single_question_optimized, 
                    question, 
                    doc_id, 
                    i
                )
                futures.append(future)
            
            submission_time = time.time() - submission_start
            logger.info(f"🚀 All {len(futures)} questions submitted in {submission_time:.3f}s")
            
            # Wait for completion with a more lenient timeout
            try:
                timeout = max(remaining_time - 2.0, 15.0)  # Increased minimum timeout for better answers
                logger.info(f"⏱️ Waiting for completion with {timeout:.1f}s timeout")
                
                completion_start = time.time()
                answers = await asyncio.wait_for(
                    asyncio.gather(*futures, return_exceptions=True),
                    timeout=timeout
                )
                completion_time = time.time() - completion_start
                
                logger.info(f"✅ All questions completed in {completion_time:.2f}s")
                
            except asyncio.TimeoutError:
                logger.error("❌ TIMEOUT: Question processing exceeded time limit")
                
                # Cancel remaining futures
                cancelled_count = 0
                for future in futures:
                    if future.cancel():
                        cancelled_count += 1
                
                logger.warning(f"⚠️ Cancelled {cancelled_count} remaining tasks")
                
                # Return timeout message for all questions
                return {
                    "answers": ["Processing timed out - please try again"] * len(request.questions)
                }
        
        # Phase 3: Process and validate results
        validation_start = time.time()
        processed_answers = []
        
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                logger.error(f"❌ Q{i+1} failed: {str(answer)}")
                processed_answers.append("Unable to process this question due to an error.")
            else:
                # Quick validation and cleaning
                clean_answer = str(answer).replace('\n', ' ').replace('\r', ' ').strip()
                if not clean_answer or clean_answer.isspace():
                    clean_answer = "Information not available in the provided document."
                processed_answers.append(clean_answer)
        
        validation_time = time.time() - validation_start
        questions_processing_time = time.time() - questions_start_time
        total_processing_time = time.time() - overall_start_time
        
        # Final logging
        logger.info(f"🎯 PERFORMANCE SUMMARY:")
        logger.info(f"   📄 Document processing: {doc_processing_time:.2f}s")
        logger.info(f"   ⚡ Questions processing: {questions_processing_time:.2f}s")
        logger.info(f"   ✅ Validation: {validation_time:.3f}s")
        logger.info(f"   🏁 TOTAL: {total_processing_time:.2f}s")
        
        # Success/failure check
        if total_processing_time <= 30.0:
            logger.info(f"🏆 SUCCESS: Completed in {total_processing_time:.2f}s (under 30s limit)")
        else:
            logger.warning(f"⚠️ OVER LIMIT: Took {total_processing_time:.2f}s (exceeded 30s)")
        
        # Return optimized response
        return {"answers": processed_answers}
        
    except Exception as e:
        processing_time = time.time() - overall_start_time
        logger.error(f"💥 CRITICAL ERROR after {processing_time:.2f}s: {str(e)}")
        traceback.print_exc()
        
        # Return error response
        return {
            "answers": ["Critical system error occurred"] * len(request.questions)
        }

@router.post("/webhook", response_model=WebhookResponse)
async def webhook_handler(request: WebhookRequest, bg_tasks: BackgroundTasks):
    """Webhook endpoint for external integrations"""
    
    try:
        # Log webhook event
        bg_tasks.add_task(log_webhook_event, request.dict())
        logger.info(f"Webhook received: {request.event_type}")
        
        return WebhookResponse(
            status="success",
            message="Webhook processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Webhook failed: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with service status"""
    
    try:
        # Quick service check
        _, embedding_service, llm_service = get_services()
        
        # Test embedding service
        test_embeddings = embedding_service.get_embeddings(["health check"])
        embedding_healthy = len(test_embeddings) > 0 and len(test_embeddings[0]) > 0
        
        status = "healthy" if embedding_healthy else "degraded"
        
        return HealthResponse(
            status=status,
            service="Ultra-Fast Document Query System",
            version=settings.api_version
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            service="Ultra-Fast Document Query System",
            version=settings.api_version
        )

async def log_webhook_event(webhook_data: Dict[str, Any]):
    """Background task for webhook logging"""
    logger.info(f"Webhook logged: {webhook_data}")

# Pre-warm endpoint for testing
@router.post("/prewarm")
async def prewarm_services():
    """Endpoint to pre-warm services"""
    try:
        start_time = time.time()
        
        # Initialize services
        doc_processor, embedding_service, llm_service = get_services()
        
        # Test operations
        test_embeddings = embedding_service.get_embeddings(["test warmup"])
        test_answer = llm_service.generate_answer("What is this?", ["This is a test."])
        
        warmup_time = time.time() - start_time
        
        return {
            "status": "success",
            "warmup_time": round(warmup_time, 2),
            "message": "Services pre-warmed successfully"
        }
        
    except Exception as e:
        logger.error(f"Pre-warm failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Pre-warm failed: {str(e)}"
        }