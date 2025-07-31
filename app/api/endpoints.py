from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
import time
from typing import Dict, Any
from app.models.request import DocumentQueryRequest, WebhookRequest
from app.models.response import QueryResponse, WebhookResponse, HealthResponse
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService  
from app.services.llm_service import LLMService
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()

# initialize services
doc_processor = DocumentProcessor()
embedding_service = EmbeddingService()
llm_service = LLMService()

@router.post("/hackrx/run", response_model=QueryResponse)
async def process_document_queries(request: DocumentQueryRequest):
    """main hackathon endpoint"""
    
    start_time = time.time()
    
    try:
        # process document
        doc_id, chunks = await doc_processor.process_document_from_url(str(request.documents))
        
        answers = []
        
        # process each question
        for question in request.questions:
            # search for relevant chunks
            search_results = embedding_service.hybrid_search(
                query=question,
                doc_id=doc_id,
                top_k=settings.max_context_chunks
            )
            
            # extract context texts
            context_chunks = [result['text'] for result in search_results]
            
            # generate answer using llm
            result = llm_service.generate_answer(question, context_chunks)
            
            # quality check
            if result['confidence'] >= settings.confidence_threshold:
                answers.append(result['answer'])
            else:
                # fallback: use more context
                extended_results = embedding_service.hybrid_search(
                    query=question,
                    doc_id=doc_id,
                    top_k=settings.max_context_chunks * 2
                )
                extended_context = [r['text'] for r in extended_results]
                fallback_result = llm_service.generate_answer(question, extended_context)
                answers.append(fallback_result['answer'])
        
        processing_time = time.time() - start_time
        
        logger.info(f"processed {len(request.questions)} questions in {processing_time:.2f}s")
        
        return QueryResponse(
            answers=answers,
            processing_time=processing_time,
            metadata={
                'document_id': doc_id,
                'chunks_processed': len(chunks),
                'questions_count': len(request.questions)
            }
        )
        
    except Exception as e:
        logger.error(f"query processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"processing failed: {str(e)}"
        )

@router.post("/webhook", response_model=WebhookResponse)
async def webhook_handler(request: WebhookRequest, bg_tasks: BackgroundTasks):
    """webhook endpoint for external integrations"""
    
    try:
        # log webhook event
        bg_tasks.add_task(log_webhook_event, request.dict())
        
        logger.info(f"webhook received: {request.event_type}")
        
        return WebhookResponse(
            status="success",
            message="webhook processed successfully"
        )
        
    except Exception as e:
        logger.error(f"webhook processing failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"webhook failed: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """health check endpoint"""
    
    return HealthResponse(
        status="healthy",
        service="intelligent doc query system",
        version=settings.api_version
    )

async def log_webhook_event(webhook_data: Dict[str, Any]):
    """background task for webhook logging"""
    logger.info(f"webhook logged: {webhook_data}")
