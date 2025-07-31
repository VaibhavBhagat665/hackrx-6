import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from typing import List, Dict, Any
import ssl
import urllib3
from app.core.config import settings
from app.core.logging import logger

class EmbeddingService:
    """advanced embedding and vector search service"""
    
    def __init__(self):
        # load models with SSL handling
        self.embedding_model = self._load_embedding_model()
        self.reranker = self._load_reranker_model()
        
        # init pinecone with new API
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.pinecone_index)
        
        logger.info("embedding service initialized")
    
    def _load_embedding_model(self):
        """Load embedding model with fallback strategies"""
        models_to_try = [
            settings.embedding_model,  # Primary model from settings
            'all-MiniLM-L6-v2',       # Smaller, faster alternative
            'distilbert-base-nli-stsb-mean-tokens'  # Another fallback
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load embedding model: {model_name}")
                
                # First try loading from cache (offline mode)
                try:
                    model = SentenceTransformer(model_name, local_files_only=True)
                    logger.info(f"Successfully loaded {model_name} from cache")
                    return model
                except Exception as cache_error:
                    logger.warning(f"Cache load failed for {model_name}: {cache_error}")
                
                # If cache fails, download with SSL bypass
                logger.info(f"Downloading {model_name} from Hugging Face...")
                model = SentenceTransformer(model_name)
                logger.info(f"Successfully downloaded and loaded: {model_name}")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
                continue
        
        raise Exception("Could not load any embedding model. Check your internet connection and model names.")
    
    def _load_reranker_model(self):
        """Load reranker model with fallback strategies"""
        try:
            logger.info(f"Attempting to load reranker model: {settings.rerank_model}")
            
            # First try loading from cache
            try:
                model = CrossEncoder(settings.rerank_model, local_files_only=True)
                logger.info(f"Successfully loaded reranker from cache")
                return model
            except Exception as cache_error:
                logger.warning(f"Reranker cache load failed: {cache_error}")
            
            # If cache fails, download
            logger.info("Downloading reranker model...")
            model = CrossEncoder(settings.rerank_model)
            logger.info("Successfully loaded reranker model")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load reranker model: {str(e)}")
            logger.warning("Continuing without reranker - search quality may be reduced")
            return None
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """create embeddings for text list"""
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"embedding creation failed: {str(e)}")
            raise
    
    def store_document_vectors(self, doc_id: str, chunks: List[Dict]) -> bool:
        """store document chunks as vectors"""
        try:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.create_embeddings(texts)
            
            # prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{doc_id}_{chunk['chunk_id']}"
                metadata = {
                    'text': chunk['text'],
                    'doc_id': doc_id,
                    'chunk_id': chunk['chunk_id'],
                    'word_count': chunk['word_count']
                }
                
                vectors.append((vector_id, embedding.tolist(), metadata))
            
            # batch upsert
            self.index.upsert(vectors)
            logger.info(f"stored {len(vectors)} vectors for doc {doc_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"vector storage failed: {str(e)}")
            return False
    
    def hybrid_search(self, query: str, doc_id: str = None, top_k: int = 10) -> List[Dict]:
        """hybrid search with reranking"""
        try:
            # create query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # vector search
            filter_dict = {"doc_id": doc_id} if doc_id else None
            
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # get more candidates for reranking
                include_metadata=True,
                filter=filter_dict
            )
            
            # prepare candidates for reranking
            candidates = []
            for match in search_results.matches:
                candidates.append({
                    'id': match.id,
                    'text': match.metadata['text'],
                    'vector_score': match.score,
                    'metadata': match.metadata
                })
            
            # rerank candidates if reranker is available
            if candidates and self.reranker is not None:
                try:
                    query_text_pairs = [(query, candidate['text']) for candidate in candidates]
                    rerank_scores = self.reranker.predict(query_text_pairs)
                    
                    # combine scores
                    for i, candidate in enumerate(candidates):
                        candidate['rerank_score'] = float(rerank_scores[i])
                        candidate['final_score'] = (
                            0.3 * candidate['vector_score'] + 
                            0.7 * candidate['rerank_score']
                        )
                    
                    # sort by final score
                    candidates.sort(key=lambda x: x['final_score'], reverse=True)
                    
                except Exception as rerank_error:
                    logger.warning(f"Reranking failed, using vector scores only: {rerank_error}")
                    # Fallback to vector scores only
                    for candidate in candidates:
                        candidate['final_score'] = candidate['vector_score']
                    candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            elif candidates:
                # No reranker available, use vector scores only
                for candidate in candidates:
                    candidate['final_score'] = candidate['vector_score']
                candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"search failed: {str(e)}")
            return []