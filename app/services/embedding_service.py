# app/services/embedding_service.py - Updated with local storage fallback

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
import os
from app.core.config import settings
from app.core.logging import logger

class EmbeddingService:
    """Embedding service with Pinecone or local storage fallback"""
    
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)
        self.use_pinecone = bool(settings.pinecone_api_key and settings.pinecone_env)
        
        if self.use_pinecone:
            try:
                import pinecone
                pinecone.init(
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_env
                )
                self.index = pinecone.Index(settings.pinecone_index_name)
                logger.info("Pinecone initialized successfully")
            except Exception as e:
                logger.warning(f"Pinecone initialization failed: {e}. Using local storage.")
                self.use_pinecone = False
                self._init_local_storage()
        else:
            logger.info("Using local vector storage (Pinecone not configured)")
            self._init_local_storage()
    
    def _init_local_storage(self):
        """Initialize local storage for vectors"""
        self.local_storage_path = "data/vectors"
        os.makedirs(self.local_storage_path, exist_ok=True)
        self.vectors_cache = {}
    
    def store_document_vectors(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Store document vectors using Pinecone or local storage"""
        try:
            if self.use_pinecone:
                return self._store_pinecone(doc_id, chunks)
            else:
                return self._store_local(doc_id, chunks)
        except Exception as e:
            logger.error(f"Vector storage failed: {e}")
            return False
    
    def _store_pinecone(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Store vectors in Pinecone"""
        vectors_to_upsert = []
        
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.model.encode(chunk['text']).tolist()
            
            # Create vector record
            vector_id = f"{doc_id}_{i}"
            vector_record = {
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'text': chunk['text'][:1000],  # Limit metadata size
                    'source_url': chunk.get('source_url', '')
                }
            }
            vectors_to_upsert.append(vector_record)
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Stored {len(vectors_to_upsert)} vectors in Pinecone for doc {doc_id}")
        return True
    
    def _store_local(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Store vectors locally"""
        doc_vectors = []
        
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.model.encode(chunk['text'])
            
            # Create vector record
            vector_record = {
                'id': f"{doc_id}_{i}",
                'embedding': embedding.tolist(),
                'metadata': {
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'text': chunk['text'],
                    'source_url': chunk.get('source_url', '')
                }
            }
            doc_vectors.append(vector_record)
        
        # Store in memory cache
        self.vectors_cache[doc_id] = doc_vectors
        
        # Also save to disk
        file_path = os.path.join(self.local_storage_path, f"{doc_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(doc_vectors, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Stored {len(doc_vectors)} vectors locally for doc {doc_id}")
        return True
    
    def hybrid_search(self, query: str, doc_id: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search using Pinecone or local storage"""
        try:
            if self.use_pinecone:
                return self._search_pinecone(query, doc_id, top_k)
            else:
                return self._search_local(query, doc_id, top_k)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _search_pinecone(self, query: str, doc_id: str, top_k: int) -> List[Dict]:
        """Search using Pinecone"""
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter={"doc_id": {"$eq": doc_id}},
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'text': match['metadata']['text'],
                'score': match['score'],
                'chunk_index': match['metadata']['chunk_index']
            })
        
        return formatted_results
    
    def _search_local(self, query: str, doc_id: str, top_k: int) -> List[Dict]:
        """Search using local storage"""
        # Get document vectors
        doc_vectors = self.vectors_cache.get(doc_id)
        if not doc_vectors:
            # Try loading from disk
            file_path = os.path.join(self.local_storage_path, f"{doc_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_vectors = json.load(f)
                self.vectors_cache[doc_id] = doc_vectors
            else:
                logger.warning(f"No vectors found for document {doc_id}")
                return []
        
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        similarities = []
        for vector_record in doc_vectors:
            chunk_embedding = np.array(vector_record['embedding'])
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append({
                'text': vector_record['metadata']['text'],
                'score': float(similarity),
                'chunk_index': vector_record['metadata']['chunk_index']
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]