import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import json
import os
import logging
import threading
import time
import hashlib
import pickle
import gzip
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Enhanced embedding service with better accuracy and hybrid search capabilities"""
    
    def __init__(self):
        self.use_pinecone = bool(settings.pinecone_api_key and settings.pinecone_env)
        
        # Thread-local storage for model safety
        self._local = threading.local()
        
        # Enhanced caching system
        self._embedding_cache = {}
        self._cache_metadata = {}
        self._cache_lock = threading.Lock()
        
        # Initialize enhanced embedding model
        try:
            logger.info(f"Loading enhanced embedding model: {settings.embedding_model}")
            self.model = SentenceTransformer(settings.embedding_model)
            
            # Advanced model warming with diverse examples
            warmup_texts = [
                "This is a technical document about software development.",
                "Financial reports and quarterly earnings data.",
                "Research methodology and experimental results.",
                "Legal contract terms and conditions.",
                "Medical diagnosis and treatment procedures."
            ]
            self.model.encode(warmup_texts, show_progress_bar=False)
            logger.info("Enhanced embedding model loaded and warmed up successfully")
            
        except Exception as e:
            logger.error(f"Failed to load primary model: {str(e)}")
            try:
                logger.info("Attempting fallback to all-MiniLM-L6-v2")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                warmup_texts = ["fallback test"]
                self.model.encode(warmup_texts, show_progress_bar=False)
                logger.info("Fallback embedding model loaded successfully")
            except Exception as e2:
                logger.error(f"All embedding models failed: {str(e2)}")
                raise Exception(f"Could not load any embedding model: {str(e2)}")
        
        # Initialize storage backend
        if self.use_pinecone:
            self._init_pinecone()
        else:
            self._init_enhanced_local_storage()
    
    def _init_pinecone(self):
        """Initialize Pinecone with enhanced configuration"""
        try:
            import pinecone
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_env
            )
            self.index = pinecone.Index(settings.pinecone_index_name)
            logger.info("Enhanced Pinecone initialized successfully")
        except Exception as e:
            logger.warning(f"Pinecone initialization failed: {e}. Using enhanced local storage.")
            self.use_pinecone = False
            self._init_enhanced_local_storage()
    
    def _init_enhanced_local_storage(self):
        """Initialize enhanced local storage with compression and indexing"""
        self.local_storage_path = "data/vectors"
        self.index_storage_path = "data/indexes"
        os.makedirs(self.local_storage_path, exist_ok=True)
        os.makedirs(self.index_storage_path, exist_ok=True)
        
        self.vectors_cache = {}
        self.vectors_index_cache = {}
        self.vectors_cache_lock = threading.Lock()
        
        logger.info(f"Enhanced local vector storage initialized at: {self.local_storage_path}")
    
    def get_embeddings_enhanced(self, texts: List[str]) -> List[List[float]]:
        """Enhanced embedding generation with intelligent caching and batch optimization"""
        try:
            if not texts:
                return []
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Advanced cache checking with content hashing
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            with self._cache_lock:
                for i, text in enumerate(texts):
                    # Use content hash for better cache key
                    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                    
                    if text_hash in self._embedding_cache:
                        cached_embeddings.append((i, self._embedding_cache[text_hash]))
                        # Update cache metadata
                        self._cache_metadata[text_hash] = time.time()
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            
            logger.info(f"Cache hit: {len(cached_embeddings)}, Cache miss: {len(uncached_texts)}")
            
            # Generate embeddings for uncached texts with optimized batching
            new_embeddings = []
            if uncached_texts:
                batch_size = min(settings.embedding_batch_size, len(uncached_texts))
                
                # Process in optimized batches
                for batch_start in range(0, len(uncached_texts), batch_size):
                    batch_end = min(batch_start + batch_size, len(uncached_texts))
                    batch_texts = uncached_texts[batch_start:batch_end]
                    
                    # Generate embeddings with enhanced parameters
                    embeddings = self.model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        batch_size=batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # Better for cosine similarity
                    )
                    
                    # Cache and store results
                    with self._cache_lock:
                        for local_i, (text, embedding) in enumerate(zip(batch_texts, embeddings)):
                            global_i = uncached_indices[batch_start + local_i]
                            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                            
                            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                            self._embedding_cache[text_hash] = embedding_list
                            self._cache_metadata[text_hash] = time.time()
                            new_embeddings.append((global_i, embedding_list))
                
                # Intelligent cache management
                self._manage_cache()
            
            # Combine and order results
            all_embeddings = [None] * len(texts)
            for index, embedding in cached_embeddings + new_embeddings:
                all_embeddings[index] = embedding
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            return all_embeddings
                
        except Exception as e:
            logger.error(f"Enhanced embedding generation failed: {str(e)}")
            # Return fallback embeddings
            return [[0.0] * settings.embedding_dimension for _ in texts]
    
    def _manage_cache(self):
        """Intelligent cache management based on usage and time"""
        max_cache_size = settings.cache_size
        
        if len(self._embedding_cache) > max_cache_size:
            current_time = time.time()
            
            # Sort by last access time and remove oldest 30%
            sorted_items = sorted(
                self._cache_metadata.items(),
                key=lambda x: x[1]
            )
            
            items_to_remove = int(len(sorted_items) * 0.3)
            for text_hash, _ in sorted_items[:items_to_remove]:
                self._embedding_cache.pop(text_hash, None)
                self._cache_metadata.pop(text_hash, None)
            
            logger.info(f"Cache cleaned: removed {items_to_remove} old entries")
    
    def store_document_vectors_enhanced(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Enhanced vector storage with compression and indexing"""
        try:
            logger.info(f"Storing enhanced vectors for document {doc_id} with {len(chunks)} chunks")
            
            if self.use_pinecone:
                return self._store_pinecone_enhanced(doc_id, chunks)
            else:
                return self._store_local_enhanced(doc_id, chunks)
                
        except Exception as e:
            logger.error(f"Enhanced vector storage failed for doc {doc_id}: {str(e)}")
            return False
    
    def _store_local_enhanced(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Enhanced local storage with compression and fast indexing"""
        try:
            doc_vectors = []
            doc_index = []
            
            # Extract texts for batch embedding
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.get_embeddings_enhanced(texts)
            
            # Build enhanced vector records
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_record = {
                    'id': f"{doc_id}_{i}",
                    'embedding': embedding,
                    'metadata': {
                        'doc_id': doc_id,
                        'chunk_index': i,
                        'text': chunk['text'],
                        'source_url': chunk.get('source_url', ''),
                        'quality_score': chunk.get('chunk_quality_score', 0.5),
                        'enhanced_features': chunk.get('enhanced_features', {}),
                        'word_count': chunk.get('word_count', 0)
                    }
                }
                doc_vectors.append(vector_record)
                
                # Create search index entry
                index_entry = {
                    'chunk_id': f"{doc_id}_{i}",
                    'text_preview': chunk['text'][:200],
                    'quality_score': chunk.get('chunk_quality_score', 0.5),
                    'word_count': chunk.get('word_count', 0)
                }
                doc_index.append(index_entry)
            
            # Store in memory cache with thread safety
            with self.vectors_cache_lock:
                self.vectors_cache[doc_id] = doc_vectors
                self.vectors_index_cache[doc_id] = doc_index
            
            # Enhanced disk storage with compression
            try:
                # Store vectors with compression
                if settings.enable_compression:
                    vectors_path = os.path.join(self.local_storage_path, f"{doc_id}.pkl.gz")
                    with gzip.open(vectors_path, 'wb') as f:
                        pickle.dump(doc_vectors, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    vectors_path = os.path.join(self.local_storage_path, f"{doc_id}.json")
                    with open(vectors_path, 'w', encoding='utf-8') as f:
                        json.dump(doc_vectors, f, ensure_ascii=False)
                
                # Store search index
                index_path = os.path.join(self.index_storage_path, f"{doc_id}_index.json")
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(doc_index, f, ensure_ascii=False)
                
                logger.info(f"Enhanced vectors and index saved for doc {doc_id}")
                
            except Exception as disk_error:
                logger.warning(f"Disk storage failed: {disk_error}, kept in memory")
            
            logger.info(f"Stored {len(doc_vectors)} enhanced vectors for doc {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced local storage failed: {str(e)}")
            return False
    
    def _store_pinecone_enhanced(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Enhanced Pinecone storage with metadata optimization"""
        try:
            vectors_to_upsert = []
            
            # Extract texts for batch embedding
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.get_embeddings_enhanced(texts)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_record = {
                    'id': f"{doc_id}_{i}",
                    'values': embedding,
                    'metadata': {
                        'doc_id': doc_id,
                        'chunk_index': i,
                        'text': chunk['text'][:1000],  # Increased metadata size
                        'source_url': chunk.get('source_url', ''),
                        'quality_score': chunk.get('chunk_quality_score', 0.5),
                        'word_count': chunk.get('word_count', 0),
                        'has_numbers': chunk.get('enhanced_features', {}).get('has_numbers', False),
                        'has_dates': chunk.get('enhanced_features', {}).get('has_dates', False)
                    }
                }
                vectors_to_upsert.append(vector_record)
            
            # Enhanced batch upsert with progress tracking
            batch_size = 100
            total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(vectors_to_upsert))
                batch = vectors_to_upsert[start_idx:end_idx]
                
                self.index.upsert(vectors=batch)
                logger.info(f"Uploaded batch {batch_num + 1}/{total_batches}")
            
            logger.info(f"Stored {len(vectors_to_upsert)} enhanced vectors in Pinecone for doc {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced Pinecone storage failed: {str(e)}")
            return False
    
    def hybrid_search_enhanced(self, query: str, doc_id: str, top_k: int = 6) -> List[Dict]:
        """Enhanced hybrid search combining semantic and keyword matching"""
        try:
            logger.info(f"Enhanced hybrid search for: '{query[:50]}...' in doc {doc_id}")
            
            if self.use_pinecone:
                results = self._search_pinecone_enhanced(query, doc_id, top_k)
            else:
                results = self._search_local_enhanced(query, doc_id, top_k)
            
            # Apply enhanced reranking if enabled
            if settings.rerank_enabled and results:
                results = self._rerank_results_enhanced(query, results)
            
            logger.info(f"Enhanced search returned {len(results)} high-quality results")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced search failed for query '{query[:50]}...': {str(e)}")
            return []
    
    def _search_local_enhanced(self, query: str, doc_id: str, top_k: int) -> List[Dict]:
        """Enhanced local search with hybrid semantic-keyword matching"""
        try:
            # Load vectors from cache or disk
            doc_vectors = self._load_vectors_optimized(doc_id)
            
            if not doc_vectors:
                logger.error(f"No vectors found for document {doc_id}")
                return []
            
            # Generate query embedding
            query_embeddings = self.get_embeddings_enhanced([query])
            if not query_embeddings or not query_embeddings[0]:
                logger.error("Failed to generate query embedding")
                return []
            
            query_embedding = np.array(query_embeddings[0])
            
            # Enhanced similarity calculation with hybrid scoring
            results = []
            
            # Prepare embeddings matrix for vectorized operations
            embeddings_matrix = np.array([v['embedding'] for v in doc_vectors])
            
            # Semantic similarity (cosine similarity)
            semantic_scores = cosine_similarity([query_embedding], embeddings_matrix)[0]
            
            # Keyword similarity scoring
            query_keywords = set(query.lower().split())
            
            for i, (vector_record, semantic_score) in enumerate(zip(doc_vectors, semantic_scores)):
                text = vector_record['metadata']['text'].lower()
                text_words = set(text.split())
                
                # Calculate keyword overlap
                keyword_overlap = len(query_keywords.intersection(text_words))
                keyword_score = keyword_overlap / max(1, len(query_keywords))
                
                # Hybrid scoring
                if settings.hybrid_search_enabled:
                    final_score = (
                        settings.semantic_search_weight * semantic_score +
                        settings.keyword_search_weight * keyword_score
                    )
                else:
                    final_score = semantic_score
                
                # Quality boost
                quality_score = vector_record['metadata'].get('quality_score', 0.5)
                final_score *= (0.8 + 0.2 * quality_score)  # 10-20% boost based on quality
                
                if final_score >= settings.similarity_threshold:
                    results.append({
                        'text': vector_record['metadata']['text'],
                        'score': float(final_score),
                        'semantic_score': float(semantic_score),
                        'keyword_score': float(keyword_score),
                        'quality_score': quality_score,
                        'chunk_index': vector_record['metadata']['chunk_index'],
                        'word_count': vector_record['metadata'].get('word_count', 0)
                    })
            
            # Sort by combined score and return top results
            results.sort(key=lambda x: x['score'], reverse=True)
            final_results = results[:top_k]
            
            logger.info(f"Enhanced local search found {len(final_results)} high-quality results")
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced local search failed: {str(e)}")
            return []
    
    def _load_vectors_optimized(self, doc_id: str) -> Optional[List[Dict]]:
        """Optimized vector loading with memory and disk caching"""
        # Check memory cache first
        with self.vectors_cache_lock:
            if doc_id in self.vectors_cache:
                return self.vectors_cache[doc_id]
        
        # Load from disk
        try:
            # Try compressed format first
            if settings.enable_compression:
                vectors_path = os.path.join(self.local_storage_path, f"{doc_id}.pkl.gz")
                if os.path.exists(vectors_path):
                    with gzip.open(vectors_path, 'rb') as f:
                        doc_vectors = pickle.load(f)
                else:
                    # Fallback to JSON
                    vectors_path = os.path.join(self.local_storage_path, f"{doc_id}.json")
                    if os.path.exists(vectors_path):
                        with open(vectors_path, 'r', encoding='utf-8') as f:
                            doc_vectors = json.load(f)
                    else:
                        return None
            else:
                vectors_path = os.path.join(self.local_storage_path, f"{doc_id}.json")
                if os.path.exists(vectors_path):
                    with open(vectors_path, 'r', encoding='utf-8') as f:
                        doc_vectors = json.load(f)
                else:
                    return None
            
            # Cache in memory
            with self.vectors_cache_lock:
                self.vectors_cache[doc_id] = doc_vectors
            
            logger.info(f"Loaded {len(doc_vectors)} vectors from disk for doc {doc_id}")
            return doc_vectors
            
        except Exception as e:
            logger.error(f"Failed to load vectors for doc {doc_id}: {str(e)}")
            return None
    
    def _rerank_results_enhanced(self, query: str, results: List[Dict]) -> List[Dict]:
        """Enhanced result reranking for better accuracy"""
        try:
            # Advanced reranking based on multiple factors
            query_words = query.lower().split()
            
            for result in results:
                text = result['text'].lower()
                
                # Position scoring (earlier mentions get higher scores)
                position_scores = []
                for word in query_words:
                    pos = text.find(word)
                    if pos != -1:
                        # Earlier positions get higher scores
                        position_score = max(0, 1.0 - (pos / len(text)))
                        position_scores.append(position_score)
                
                avg_position_score = sum(position_scores) / max(1, len(position_scores))
                
                # Density scoring (query words close together)
                word_positions = []
                for word in query_words:
                    pos = text.find(word)
                    if pos != -1:
                        word_positions.append(pos)
                
                density_score = 0.5
                if len(word_positions) > 1:
                    word_span = max(word_positions) - min(word_positions)
                    density_score = max(0.1, 1.0 - (word_span / len(text)))
                
                # Enhanced final scoring
                enhanced_score = (
                    result['score'] * 0.6 +  # Original semantic score
                    avg_position_score * 0.2 +  # Position bonus
                    density_score * 0.1 +  # Density bonus
                    result.get('quality_score', 0.5) * 0.1  # Quality bonus
                )
                
                result['enhanced_score'] = enhanced_score
            
            # Re-sort by enhanced score
            results.sort(key=lambda x: x.get('enhanced_score', x['score']), reverse=True)
            
            logger.info("Results reranked with enhanced scoring")
            return results
            
        except Exception as e:
            logger.warning(f"Enhanced reranking failed: {str(e)}, using original order")
            return results
    
    def _search_pinecone_enhanced(self, query: str, doc_id: str, top_k: int) -> List[Dict]:
        """Enhanced Pinecone search with metadata filtering"""
        try:
            # Generate query embedding
            query_embeddings = self.get_embeddings_enhanced([query])
            if not query_embeddings or not query_embeddings[0]:
                logger.error("Failed to generate query embedding")
                return []
            
            query_embedding = query_embeddings[0]
            
            # Enhanced search with metadata filters
            filter_dict = {"doc_id": {"$eq": doc_id}}
            
            # Add quality filter for better results
            if settings.similarity_threshold > 0.5:
                filter_dict["quality_score"] = {"$gte": 0.3}
            
            results = self.index.query(
                vector=query_embedding,
                top_k=min(top_k * 2, 20),  # Get more results for reranking
                filter=filter_dict,
                include_metadata=True
            )
            
            # Format enhanced results
            formatted_results = []
            for match in results['matches']:
                metadata = match['metadata']
                formatted_results.append({
                    'text': metadata['text'],
                    'score': match['score'],
                    'chunk_index': metadata['chunk_index'],
                    'quality_score': metadata.get('quality_score', 0.5),
                    'word_count': metadata.get('word_count', 0),
                    'has_numbers': metadata.get('has_numbers', False),
                    'has_dates': metadata.get('has_dates', False)
                })
            
            return formatted_results[:top_k]
            
        except Exception as e:
            logger.error(f"Enhanced Pinecone search failed: {str(e)}")
            return []
    
    def get_document_stats(self, doc_id: str) -> Optional[Dict]:
        """Get comprehensive statistics for a processed document"""
        try:
            with self.vectors_cache_lock:
                if doc_id in self.vectors_cache:
                    vectors = self.vectors_cache[doc_id]
                    
                    stats = {
                        'total_chunks': len(vectors),
                        'avg_quality_score': sum(v['metadata'].get('quality_score', 0.5) for v in vectors) / len(vectors),
                        'total_words': sum(v['metadata'].get('word_count', 0) for v in vectors),
                        'chunks_with_numbers': sum(1 for v in vectors if v['metadata'].get('has_numbers', False)),
                        'chunks_with_dates': sum(1 for v in vectors if v['metadata'].get('has_dates', False)),
                        'embedding_dimension': len(vectors[0]['embedding']) if vectors else 0
                    }
                    
                    return stats
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document stats: {str(e)}")
            return None
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)
            
            with self._cache_lock:
                expired_keys = [
                    key for key, timestamp in self._cache_metadata.items()
                    if timestamp < cutoff_time
                ]
                
                for key in expired_keys:
                    self._embedding_cache.pop(key, None)
                    self._cache_metadata.pop(key, None)
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Wrapper method for backward compatibility"""
        return self.get_embeddings_enhanced(texts)
    
    def store_document_vectors(self, doc_id: str, chunks: List[Dict]) -> bool:
        """Wrapper method for backward compatibility"""
        return self.store_document_vectors_enhanced(doc_id, chunks)
    
    def hybrid_search(self, query: str, doc_id: str, top_k: int = 6) -> List[Dict]:
        """Wrapper method for backward compatibility"""
        return self.hybrid_search_enhanced(query, doc_id, top_k)
    
    def __del__(self):
        """Enhanced cleanup with comprehensive resource management"""
        try:
            # Clear all caches
            with self._cache_lock:
                self._embedding_cache.clear()
                self._cache_metadata.clear()
                
            if hasattr(self, 'vectors_cache_lock'):
                with self.vectors_cache_lock:
                    self.vectors_cache.clear()
                    self.vectors_index_cache.clear()
                    
            # Close HTTP client
            if hasattr(self._local, 'client'):
                self._local.client.close()
                
            logger.info("Enhanced embedding service cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")
            pass