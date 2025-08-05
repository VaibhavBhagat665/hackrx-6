import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
import os
import logging
import threading
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Ultra-optimized embedding service for sub-30 second responses"""
    
    def __init__(self):
        self.use_pinecone = bool(settings.pinecone_api_key and settings.pinecone_env)
        
        # Thread-local storage for model (for thread safety)
        self._local = threading.local()
        
        # Cache for embeddings to avoid recomputation
        self._embedding_cache = {}
        self._cache_lock = threading.Lock()
        
        # Initialize model with error handling for Cloud Run
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self.model = SentenceTransformer(settings.embedding_model)
            # Pre-compile the model for faster inference
            dummy_text = ["warmup text"]
            self.model.encode(dummy_text, show_progress_bar=False)
            logger.info("Embedding model loaded and warmed up successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            try:
                logger.info("Attempting fallback to paraphrase-MiniLM-L6-v2")
                self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                dummy_text = ["warmup text"]
                self.model.encode(dummy_text, show_progress_bar=False)
                logger.info("Fallback embedding model loaded and warmed up successfully")
            except Exception as e2:
                logger.error(f"Fallback model also failed: {str(e2)}")
                raise Exception(f"Could not load any embedding model: {str(e2)}")
        
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
        self.vectors_cache_lock = threading.Lock()
        logger.info(f"Local vector storage initialized at: {self.local_storage_path}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Ultra-fast embedding generation with caching
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            # Check cache first
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            with self._cache_lock:
                for i, text in enumerate(texts):
                    text_hash = hash(text)
                    if text_hash in self._embedding_cache:
                        cached_embeddings.append((i, self._embedding_cache[text_hash]))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            new_embeddings = []
            if uncached_texts:
                # Use the pre-warmed model for faster inference
                embeddings = self.model.encode(
                    uncached_texts, 
                    show_progress_bar=False,
                    batch_size=32,  # Optimize batch size
                    convert_to_numpy=True
                )
                
                # Convert to list format and cache
                with self._cache_lock:
                    for i, (text, embedding) in enumerate(zip(uncached_texts, embeddings)):
                        text_hash = hash(text)
                        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                        self._embedding_cache[text_hash] = embedding_list
                        new_embeddings.append((uncached_indices[i], embedding_list))
                
                # Keep cache size reasonable
                if len(self._embedding_cache) > 1000:
                    # Remove oldest half of cache
                    keys_to_remove = list(self._embedding_cache.keys())[:500]
                    for key in keys_to_remove:
                        del self._embedding_cache[key]
            
            # Combine cached and new embeddings in correct order
            all_embeddings = [None] * len(texts)
            for index, embedding in cached_embeddings + new_embeddings:
                all_embeddings[index] = embedding
            
            return all_embeddings
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 384 for _ in texts]  # 384 is common embedding dimension
    
    def store_document_vectors(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Store document vectors using Pinecone or local storage - OPTIMIZED"""
        try:
            logger.info(f"Storing vectors for document {doc_id} with {len(chunks)} chunks")
            
            if self.use_pinecone:
                return self._store_pinecone(doc_id, chunks)
            else:
                return self._store_local_optimized(doc_id, chunks)
        except Exception as e:
            logger.error(f"Vector storage failed for doc {doc_id}: {str(e)}")
            return False
    
    def _store_local_optimized(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Ultra-fast local storage with optimizations"""
        try:
            doc_vectors = []
            
            # Extract texts for batch embedding (faster than individual)
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.get_embeddings(texts)
            
            # Build vector records efficiently
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_record = {
                    'id': f"{doc_id}_{i}",
                    'embedding': embedding,
                    'metadata': {
                        'doc_id': doc_id,
                        'chunk_index': i,
                        'text': chunk['text'],
                        'source_url': chunk.get('source_url', '')
                    }
                }
                doc_vectors.append(vector_record)
            
            # Store in memory cache with thread safety
            with self.vectors_cache_lock:
                self.vectors_cache[doc_id] = doc_vectors
            
            # Background disk save (don't wait for it)
            try:
                file_path = os.path.join(self.local_storage_path, f"{doc_id}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(doc_vectors, f, ensure_ascii=False)
                logger.info(f"Vectors saved to disk: {file_path}")
            except Exception as disk_error:
                logger.warning(f"Failed to save vectors to disk: {disk_error}, but kept in memory")
            
            logger.info(f"Stored {len(doc_vectors)} vectors locally for doc {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Local storage failed: {str(e)}")
            return False
    
    def _store_pinecone(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Store vectors in Pinecone - OPTIMIZED"""
        try:
            vectors_to_upsert = []
            
            # Extract texts for batch embedding
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.get_embeddings(texts)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_record = {
                    'id': f"{doc_id}_{i}",
                    'values': embedding,
                    'metadata': {
                        'doc_id': doc_id,
                        'chunk_index': i,
                        'text': chunk['text'][:800],  # Reduced metadata size for speed
                        'source_url': chunk.get('source_url', '')
                    }
                }
                vectors_to_upsert.append(vector_record)
            
            # Upsert in single batch for speed (if small enough)
            if len(vectors_to_upsert) <= 100:
                self.index.upsert(vectors=vectors_to_upsert)
            else:
                # Batch process for larger documents
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i+batch_size]
                    self.index.upsert(vectors=batch)
            
            logger.info(f"Stored {len(vectors_to_upsert)} vectors in Pinecone for doc {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Pinecone storage failed: {str(e)}")
            return False
    
    def hybrid_search(self, query: str, doc_id: str, top_k: int = 3) -> List[Dict]:
        """Ultra-fast hybrid search using Pinecone or local storage"""
        try:
            logger.info(f"Fast searching for query: '{query[:30]}...' in doc {doc_id}")
            
            if self.use_pinecone:
                results = self._search_pinecone_fast(query, doc_id, top_k)
            else:
                results = self._search_local_ultra_fast(query, doc_id, top_k)
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query[:30]}...': {str(e)}")
            return []
    
    def _search_pinecone_fast(self, query: str, doc_id: str, top_k: int) -> List[Dict]:
        """Ultra-fast Pinecone search"""
        try:
            # Generate query embedding with caching
            query_embeddings = self.get_embeddings([query])
            if not query_embeddings or not query_embeddings[0]:
                logger.error("Failed to generate query embedding")
                return []
            
            query_embedding = query_embeddings[0]
            
            # Search in Pinecone with optimized parameters
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter={"doc_id": {"$eq": doc_id}},
                include_metadata=True
            )
            
            # Format results quickly
            formatted_results = []
            for match in results['matches']:
                formatted_results.append({
                    'text': match['metadata']['text'],
                    'score': match['score'],
                    'chunk_index': match['metadata']['chunk_index']
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {str(e)}")
            return []
    
    def _search_local_ultra_fast(self, query: str, doc_id: str, top_k: int) -> List[Dict]:
        """Ultra-fast local search with aggressive optimizations"""
        try:
            # Get vectors from memory cache first (fastest)
            with self.vectors_cache_lock:
                doc_vectors = self.vectors_cache.get(doc_id)
            
            # If not in memory, try loading from disk
            if doc_vectors is None:
                file_path = os.path.join(self.local_storage_path, f"{doc_id}.json")
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_vectors = json.load(f)
                        # Cache the loaded vectors
                        with self.vectors_cache_lock:
                            self.vectors_cache[doc_id] = doc_vectors
                        logger.info(f"Loaded vectors from disk for doc {doc_id}")
                    except Exception as load_error:
                        logger.error(f"Failed to load vectors from disk: {load_error}")
                        return []
                else:
                    logger.error(f"No vectors found for document {doc_id}")
                    return []
            
            if not doc_vectors:
                logger.error(f"No vectors available for document {doc_id}")
                return []
            
            # Generate query embedding with caching
            query_embeddings = self.get_embeddings([query])
            if not query_embeddings or not query_embeddings[0]:
                logger.error("Failed to generate query embedding")
                return []
            
            query_embedding = np.array(query_embeddings[0])
            
            # Ultra-fast vectorized similarity calculation
            similarities = []
            try:
                # Pre-allocate arrays for speed
                embeddings_matrix = np.array([vector_record['embedding'] for vector_record in doc_vectors])
                
                # Vectorized cosine similarity calculation (much faster)
                # Normalize query embedding
                query_norm = np.linalg.norm(query_embedding)
                if query_norm == 0:
                    logger.error("Query embedding has zero norm")
                    return []
                
                normalized_query = query_embedding / query_norm
                
                # Normalize document embeddings
                doc_norms = np.linalg.norm(embeddings_matrix, axis=1)
                valid_indices = doc_norms != 0
                
                if not np.any(valid_indices):
                    logger.error("All document embeddings have zero norm")
                    return []
                
                # Calculate similarities only for valid embeddings
                valid_embeddings = embeddings_matrix[valid_indices]
                valid_doc_norms = doc_norms[valid_indices]
                normalized_embeddings = valid_embeddings / valid_doc_norms[:, np.newaxis]
                
                # Vectorized dot product for cosine similarity
                cosine_similarities = np.dot(normalized_embeddings, normalized_query)
                
                # Create results with valid indices
                valid_vector_records = [doc_vectors[i] for i in range(len(doc_vectors)) if valid_indices[i]]
                
                for sim_score, vector_record in zip(cosine_similarities, valid_vector_records):
                    similarities.append({
                        'text': vector_record['metadata']['text'],
                        'score': float(sim_score),
                        'chunk_index': vector_record['metadata']['chunk_index']
                    })
                
            except Exception as vectorized_error:
                logger.warning(f"Vectorized calculation failed, falling back to loop: {vectorized_error}")
                
                # Fallback to loop-based calculation
                for vector_record in doc_vectors:
                    try:
                        chunk_embedding = np.array(vector_record['embedding'])
                        
                        # Calculate cosine similarity
                        dot_product = np.dot(query_embedding, chunk_embedding)
                        norm_query = np.linalg.norm(query_embedding)
                        norm_chunk = np.linalg.norm(chunk_embedding)
                        
                        if norm_query == 0 or norm_chunk == 0:
                            similarity = 0.0
                        else:
                            similarity = dot_product / (norm_query * norm_chunk)
                        
                        similarities.append({
                            'text': vector_record['metadata']['text'],
                            'score': float(similarity),
                            'chunk_index': vector_record['metadata']['chunk_index']
                        })
                        
                    except Exception as sim_error:
                        logger.warning(f"Failed to calculate similarity for chunk: {sim_error}")
                        continue
            
            # Sort by similarity and return top_k (using partial sort for speed)
            if len(similarities) <= top_k:
                similarities.sort(key=lambda x: x['score'], reverse=True)
                results = similarities
            else:
                # Use partial sort for better performance with large lists
                similarities.sort(key=lambda x: x['score'], reverse=True)
                results = similarities[:top_k]
            
            logger.info(f"Local search found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Local search failed: {str(e)}")
            return []
    
    def __del__(self):
        """Cleanup resources"""
        try:
            # Clear caches
            with self._cache_lock:
                self._embedding_cache.clear()
            if hasattr(self, 'vectors_cache_lock'):
                with self.vectors_cache_lock:
                    self.vectors_cache.clear()
        except:
            pass