from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
import time
import traceback
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
        # Log when request is received
        logger.info(f"Request received for /hackrx/run endpoint - Document URL: {request.documents}, Questions count: {len(request.questions)}")
        
        # Before document processing
        logger.info(f"Starting document processing from URL: {request.documents}")
        doc_id, chunks = await doc_processor.process_document_from_url(str(request.documents))
        
        # After document processing
        logger.info(f"Document processing completed - Document ID: {doc_id}, Chunks created: {len(chunks)}")
        
        answers = []
        
        for i, question in enumerate(request.questions, 1):
            logger.info(f"Processing question {i}/{len(request.questions)}: {question}")
            
            # Before embedding search
            logger.info(f"Starting hybrid search for question {i} with top_k={settings.max_context_chunks}")
            search_results = embedding_service.hybrid_search(
                query=question,
                doc_id=doc_id,
                top_k=settings.max_context_chunks
            )
            
            # After getting search results
            logger.info(f"Hybrid search completed for question {i} - Retrieved {len(search_results)} results")
            
            context_chunks = [result['text'] for result in search_results]
            
            # Before LLM generation
            logger.info(f"Calling LLM service for question {i} with {len(context_chunks)} context chunks")
            result = llm_service.generate_answer(question, context_chunks)
            
            # After getting LLM result and checking confidence
            confidence = result['confidence']
            logger.info(f"LLM response generated for question {i} - Confidence: {confidence:.3f}, Threshold: {settings.confidence_threshold}")
            
            if confidence >= settings.confidence_threshold:
                logger.info(f"Question {i} confidence above threshold - Using primary answer")
                formatted_answer = result['answer']
            else:
                # When fallback reranking is triggered
                logger.info(f"Question {i} confidence below threshold - Triggering fallback with extended search (top_k={settings.max_context_chunks * 2})")
                
                extended_results = embedding_service.hybrid_search(
                    query=question,
                    doc_id=doc_id,
                    top_k=settings.max_context_chunks * 2
                )
                
                logger.info(f"Extended search completed for question {i} - Retrieved {len(extended_results)} results")
                
                extended_context = [r['text'] for r in extended_results]
                
                logger.info(f"Calling LLM service for fallback answer on question {i} with {len(extended_context)} context chunks")
                fallback_result = llm_service.generate_answer(question, extended_context)
                
                logger.info(f"Fallback answer generated for question {i} - Confidence: {fallback_result.get('confidence', 'N/A')}")
                formatted_answer = fallback_result['answer']
            
            # Log the final formatted answer for debugging
            logger.info(f"Final formatted answer for question {i}: {formatted_answer[:100]}{'...' if len(formatted_answer) > 100 else ''}")
            
            answers.append(formatted_answer)
        
        # After all questions are processed, log total time
        processing_time = time.time() - start_time
        logger.info(f"All questions processed successfully - Total questions: {len(request.questions)}, Total processing time: {processing_time:.2f}s, Average time per question: {processing_time/len(request.questions):.2f}s")
        
        # Final validation of answers format
        validated_answers = []
        for i, answer in enumerate(answers, 1):
            if '\n' in answer or len(answer.strip()) == 0:
                logger.warning(f"Answer {i} contains line breaks or is empty, applying emergency formatting")
                clean_answer = answer.replace('\n', ' ').replace('\r', ' ').strip()
                if not clean_answer:
                    clean_answer = "Information not available in the provided document."
                validated_answers.append(clean_answer)
            else:
                validated_answers.append(answer)
        
        logger.info(f"Response prepared with {len(validated_answers)} formatted answers")
        
        return QueryResponse(
            answers=validated_answers,
            processing_time=processing_time,
            metadata={
                'document_id': doc_id,
                'chunks_processed': len(chunks),
                'questions_count': len(request.questions)
            }
        )
        
    except Exception as e:
        # In the except block, include traceback.print_exc()
        logger.error(f"Exception in /hackrx/run endpoint: {str(e)}")
        traceback.print_exc()
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