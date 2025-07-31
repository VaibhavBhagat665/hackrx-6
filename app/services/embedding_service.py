import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import pinecone
from typing import List, Dict, Any
from app.core.config import settings
from app.core.logging import logger

class EmbeddingService:
    """advanced embedding and vector search service"""
    
    def __init__(self):
        # load models
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.reranker = CrossEncoder(settings.rerank_model)
        
        # init pinecone
        pinecone.init(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_env
        )
        self.index = pinecone.Index(settings.pinecone_index)
        
        logger.info("embedding service initialized")
    
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
            
            # rerank candidates
            if candidates:
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
            
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"search failed: {str(e)}")
            return []
