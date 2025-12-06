import streamlit as st
import time
import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from utils.model_handler import get_chat_client, get_embeddings_client
from langchain_core.messages import HumanMessage, AIMessage
import faiss
import pickle

# ML libraries for proper BM25 and reranking
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import torch

# Initialize chat client
@st.cache_resource
def get_llm_client():
    return get_chat_client(temperature=0.7)

# Initialize embedding client
def get_embedding_client():
    return get_embeddings_client()

# Cache reranking models to avoid reloading
@st.cache_resource(show_spinner=False)
def load_reranking_model(model_name):
    """Load and cache MS-MARCO reranking models"""
    try:
        print(f"[MODEL_LOADING_DEBUG] Starting to load reranking model: {model_name}")
        print(f"[MODEL_LOADING_DEBUG] Model type: CrossEncoder")
        model = CrossEncoder(model_name)
        print(f"[MODEL_LOADING_DEBUG] Model loaded successfully!")
        print(f"[MODEL_LOADING_DEBUG] Model details: {type(model)}")
        return model
    except Exception as e:
        print(f"[MODEL_LOADING_DEBUG] Error loading reranking model {model_name}: {e}")
        return None

# Vector database management functions
def get_vector_db_path(kb_identifier):
    """Get the file path for a knowledge base's vector database using KB ID or name"""
    # If it looks like a KB ID (8 characters, alphanumeric), use it directly
    if isinstance(kb_identifier, str) and len(kb_identifier) == 8 and kb_identifier.replace('-', '').isalnum():
        base_path = f"databases/vector_dbs/kb_{kb_identifier}"
    else:
        # Fallback to name-based path for backward compatibility
        safe_name = kb_identifier.replace(' ', '_').replace('/', '_')
        base_path = f"databases/vector_dbs/{safe_name}_Knowledge_Base"
    return base_path

def load_vector_db(kb_identifier):
    """Load vector database for a knowledge base using KB ID or name"""
    db_path = get_vector_db_path(kb_identifier)
    try:
        if os.path.exists(f"{db_path}.faiss") and os.path.exists(f"{db_path}_metadata.pkl"):
            # Load FAISS index
            index = faiss.read_index(f"{db_path}.faiss")
            # Load metadata
            with open(f"{db_path}_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            return index, metadata
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return None, None

def save_vector_db(kb_identifier, index, metadata):
    """Save vector database for a knowledge base using KB ID or name"""
    db_path = get_vector_db_path(kb_identifier)
    os.makedirs("databases/vector_dbs", exist_ok=True)
    try:
        # Save FAISS index
        faiss.write_index(index, f"{db_path}.faiss")
        # Save metadata
        with open(f"{db_path}_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        return True
    except Exception as e:
        st.error(f"Error saving vector database: {e}")
        return False

def add_documents_to_vector_db(kb_name, chunks, filename):
    """Add document chunks to a knowledge base's vector database"""
    try:
        # Get embedding client
        embedding_client = get_embedding_client()
        
        # Generate embeddings for chunks in batches of 8
        batch_size = 8
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            try:
                batch_embeddings = embedding_client.embed_documents(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                pass  # Skip failed batches
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Load existing vector DB or create new one
        index, metadata = load_vector_db(kb_name)
        
        if index is None:
            # Create new FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            metadata = {"chunks": [], "filenames": [], "chunk_ids": []}
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        start_id = index.ntotal
        index.add(embeddings_array)
        
        # Update metadata
        for i, chunk in enumerate(chunks):
            metadata["chunks"].append(chunk)
            metadata["filenames"].append(filename)
            metadata["chunk_ids"].append(start_id + i)
        
        # Save updated database
        if save_vector_db(kb_name, index, metadata):
            return True, len(chunks)
        else:
            return False, 0
            
    except Exception as e:
        st.error(f"Error adding documents to vector database: {e}")
        return False, 0

def get_chunk_from_source_kb(source_kb, chunk_id, filename):
    """Retrieve actual chunk content from source knowledge base"""
    try:
        from utils.vector_db import load_vector_db as load_vector_db_correct
        source_index, source_metadata = load_vector_db_correct(source_kb)
        
        if source_index is None or not isinstance(source_metadata, dict) or 'chunks' not in source_metadata:
            return f"Content from {filename} (chunk {chunk_id}) in {source_kb} - [Source KB not accessible]"
        
        # Find the chunk by matching chunk_id and filename
        source_filenames = source_metadata.get('filenames', [])
        source_chunk_ids = source_metadata.get('chunk_ids', [])
        source_chunks = source_metadata.get('chunks', [])
        
        for i, (src_filename, src_chunk_id) in enumerate(zip(source_filenames, source_chunk_ids)):
            if src_filename == filename and src_chunk_id == chunk_id:
                if i < len(source_chunks):
                    return source_chunks[i]
                    
        return f"Content from {filename} (chunk {chunk_id}) in {source_kb} - [Chunk not found in source]"
        
    except Exception as e:
        return f"Content from {filename} (chunk {chunk_id}) in {source_kb} - [Error: {e}]"

def generate_query_variations(original_query, num_variations=3):
    """Use LLM to generate semantically similar query variations for better retrieval"""
    try:
        client = get_llm_client()
        
        prompt = f"""Generate {num_variations} semantically similar rephrasings of this query for better document retrieval. The variations should:
- Preserve the exact semantic intent and meaning
- Use different wording, synonyms, or phrasing approaches
- Cover different ways users might ask the same question
- Be suitable for technical/aerospace document search

Original query: "{original_query}"

Respond with ONLY the {num_variations} variations, one per line, without numbers or bullets."""
        
        # Use the correct invoke method for CustomEndpointClient
        messages = [{"role": "user", "content": prompt}]
        response = client.invoke(messages)
        
        variations_text = response.content.strip()
        variations = [line.strip() for line in variations_text.split('\n') if line.strip()]
        
        # Filter out any empty variations and limit to requested number
        variations = [v for v in variations if v and len(v) > 5][:num_variations]
        
        print(f"[QUERY_EXPANSION] Original: {original_query}")
        print(f"[QUERY_EXPANSION] Generated {len(variations)} variations: {variations}")
        
        # Store variations in session state for display (safely)
        try:
            if hasattr(st, 'session_state') and hasattr(st.session_state, '__setattr__'):
                st.session_state.last_query_variations = {
                    'original': original_query,
                    'variations': variations,
                    'timestamp': time.time()
                }
        except Exception as session_error:
            print(f"[QUERY_EXPANSION] Could not store in session state: {session_error}")
        
        return variations
        
    except Exception as e:
        print(f"[QUERY_EXPANSION] Error generating variations: {e}")
        # Store error in session state (safely)
        try:
            if hasattr(st, 'session_state') and hasattr(st.session_state, '__setattr__'):
                st.session_state.last_query_variations = {
                    'original': original_query,
                    'error': str(e),
                    'timestamp': time.time()
                }
        except Exception as session_error:
            print(f"[QUERY_EXPANSION] Could not store error in session state: {session_error}")
        return []

def fuse_multi_query_results(all_results, final_k=5):
    """Fuse results from multiple queries using Reciprocal Rank Fusion (RRF)"""
    try:
        from collections import defaultdict
        
        # RRF scoring with k=60 (standard parameter)
        rrf_scores = defaultdict(float)
        seen_chunks = {}  # To store unique chunks by content hash
        
        # Group results by query and assign ranks
        query_results = {}
        current_query_id = 0
        
        for result in all_results:
            # Create a simple hash of the chunk content for deduplication
            content_hash = hash(result['chunk'][:100])  # Use first 100 chars for hash
            
            if content_hash not in seen_chunks:
                seen_chunks[content_hash] = result
            
            # Find which "query group" this result belongs to
            query_group = current_query_id // 20  # Assuming ~20 results per query
            if query_group not in query_results:
                query_results[query_group] = []
            query_results[query_group].append((content_hash, result))
            
            # Update current query id tracker
            if len(query_results[query_group]) >= 20:
                current_query_id += 20
        
        # Apply RRF scoring
        k = 60  # RRF parameter
        for query_id, results in query_results.items():
            for rank, (content_hash, result) in enumerate(results):
                rrf_scores[content_hash] += 1 / (k + rank + 1)
        
        # Sort by RRF score and take top results
        sorted_hashes = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for content_hash, rrf_score in sorted_hashes[:final_k]:
            if content_hash in seen_chunks:
                result = seen_chunks[content_hash].copy()
                result['rrf_score'] = rrf_score
                result['original_score'] = result.get('score', 0.0)
                result['score'] = rrf_score  # Use RRF score as primary score
                final_results.append(result)
        
        print(f"[RRF_FUSION] Fused {len(all_results)} total results into {len(final_results)} unique results")
        return final_results
        
    except Exception as e:
        print(f"[RRF_FUSION] Error in fusion, returning top results: {e}")
        # Fallback: just return top unique results by original score
        seen_content = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.get('score', 0), reverse=True):
            content_key = result['chunk'][:100]  # First 100 chars as key
            if content_key not in seen_content and len(unique_results) < final_k:
                seen_content.add(content_key)
                unique_results.append(result)
        return unique_results

def expand_query_and_retrieve(kb_name, original_query, top_k=5):
    """Generate query variations and retrieve with multi-query fusion"""
    try:
        print(f"[MULTI_QUERY] Starting multi-query retrieval for: {original_query}")
        
        # Get RAG settings to determine number of variations
        import streamlit as st
        rag_settings = st.session_state.get("rag_settings", {})
        num_variations = rag_settings.get("num_query_variations", 3)
        
        # Generate query variations
        query_variations = generate_query_variations(original_query, num_variations)
        
        # Always include the original query
        all_queries = [original_query] + query_variations
        
        print(f"[MULTI_QUERY] Searching with {len(all_queries)} queries total")
        
        # Retrieve with each query variation
        all_results = []
        candidates_per_query = max(10, top_k * 2)  # Get more candidates per query
        
        for i, query in enumerate(all_queries):
            print(f"[MULTI_QUERY] Query {i+1}/{len(all_queries)}: {query}")
            
            # Use the original search_vector_db function for each query
            query_results = search_vector_db_single(kb_name, query, candidates_per_query)
            
            # Add query source info to results
            for result in query_results:
                result['query_source'] = i
                result['source_query'] = query
            
            all_results.extend(query_results)
            print(f"[MULTI_QUERY] Got {len(query_results)} results from query {i+1}")
        
        # Fuse results using RRF
        final_results = fuse_multi_query_results(all_results, top_k)
        
        print(f"[MULTI_QUERY] Final result: {len(final_results)} unique chunks after RRF fusion")
        return final_results
        
    except Exception as e:
        print(f"[MULTI_QUERY] Error in multi-query retrieval: {e}")
        # Fallback to original single query
        return search_vector_db_single(kb_name, original_query, top_k)

def search_vector_db_single(kb_name, query, top_k=None):
    """Single query vector database search (original implementation)"""

def search_vector_db_single(kb_name, query, top_k=None):
    """Single query vector database search (original implementation)"""
    try:
        # Use RAG settings if available, otherwise use defaults
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'rag_settings'):
            rag_settings = st.session_state.rag_settings
            
            # Handle top_k
            if top_k is None:
                raw_top_k = rag_settings.get("top_k", 5)
                if isinstance(raw_top_k, slice):
                    top_k = 5
                elif isinstance(raw_top_k, (int, float)) and raw_top_k > 0:
                    top_k = int(raw_top_k)
                else:
                    top_k = 5
                    
            # Get other settings
            relevance_threshold = float(rag_settings.get("relevance_threshold", 0.0))
            reranking_enabled = bool(rag_settings.get("reranking_engine", False))
            reranking_model = rag_settings.get("reranking_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            hybrid_search_enabled = bool(rag_settings.get("hybrid_search", False))
            # Get BM25 weight from RAG settings
            bm25_weight = float(rag_settings.get("bm25_weight", 0.3))
            full_context_mode = bool(rag_settings.get("full_context_mode", False))
            
            # Handle top_k_reranker
            raw_top_k_reranker = rag_settings.get("top_k_reranker", top_k)
            if isinstance(raw_top_k_reranker, slice):
                top_k_reranker = top_k
            elif isinstance(raw_top_k_reranker, (int, float)) and raw_top_k_reranker > 0:
                top_k_reranker = int(raw_top_k_reranker)
            else:
                top_k_reranker = top_k
                
            # Log RAG settings being used
            print(f"[RAG_SETTINGS] Using settings - Full Context: {full_context_mode}, Hybrid: {hybrid_search_enabled}, Reranking: {reranking_enabled}")
            print(f"[RAG_SETTINGS] Top K: {top_k}, Top K Reranker: {top_k_reranker}, Threshold: {relevance_threshold}, BM25 Weight: {bm25_weight}")
            print(f"[RAG_SETTINGS] Reranking Model: {reranking_model}")
        else:
            # Default settings
            if top_k is None:
                top_k = 5
            relevance_threshold = 0.0
            reranking_enabled = False
            reranking_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            hybrid_search_enabled = False
            # Get BM25 weight 
            bm25_weight = 0.3
            full_context_mode = False
            top_k_reranker = top_k
            print("[RAG_SETTINGS] Using default RAG settings")
        
        # Ensure top_k is a reasonable integer
        if not isinstance(top_k, int) or top_k <= 0:
            top_k = 5
        
        # Clamp top_k to prevent memory issues
        top_k = min(max(top_k, 1), 100)
        top_k_reranker = min(max(top_k_reranker, 1), 50)
        
        # Load vector database
        from utils.vector_db import load_vector_db as load_vector_db_correct
        index, metadata = load_vector_db_correct(kb_name)
        
        if index is None or index.ntotal == 0:
            return []

        # Check metadata structure
        if not isinstance(metadata, dict):
            return []
        
        # Check for chunks in current KB or if this is Universal KB (which stores references)
        has_chunks = 'chunks' in metadata
        is_universal_kb = 'source_kbs' in metadata
        
        if not has_chunks and not is_universal_kb:
            return []
        
        # Full Context Mode: Return ALL chunks if enabled
        if full_context_mode:
            print("[RAG_SETTINGS] Full Context Mode enabled - returning all chunks")
            all_results = []
            total_chunks = len(metadata.get("chunks", [])) if has_chunks else len(metadata.get("filenames", []))
            
            for idx in range(min(total_chunks, 100)):  # Limit to 100 to prevent memory issues
                if is_universal_kb:
                    if idx >= len(metadata["filenames"]):
                        continue
                    filename = metadata["filenames"][idx]
                    chunk_id = metadata["chunk_ids"][idx]
                    source_kb = metadata["source_kbs"][idx]
                    chunk_text = get_chunk_from_source_kb(source_kb, chunk_id, filename)
                else:
                    if idx >= len(metadata["chunks"]):
                        continue
                    chunk_text = metadata["chunks"][idx]
                    filename = metadata["filenames"][idx] if idx < len(metadata["filenames"]) else "Unknown"
                
                all_results.append({
                    "chunk": chunk_text,
                    "filename": filename,
                    "score": 1.0,  # All chunks get max score in full context mode
                    "rank": idx + 1
                })
            
            return all_results[:top_k]  # Still respect top_k limit
            
        # Get embedding for query
        embedding_client = get_embedding_client()
        query_embedding = embedding_client.embed_documents([query])[0]
        
        query_array = np.array([query_embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_array)
        
        # Search with higher top_k for reranking if enabled
        search_k = top_k
        if reranking_enabled:
            search_k = min(top_k * 3, index.ntotal)  # Get more candidates for reranking
            print(f"[RAG_SETTINGS] Reranking enabled - searching for {search_k} candidates")
        else:
            search_k = min(top_k, index.ntotal)
            
        scores, indices = index.search(query_array, search_k)
        
        # Collect initial results (semantic search results)
        semantic_results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            
            # Handle Universal KB vs regular KB
            if is_universal_kb:
                # Universal KB case - need to get chunk from source KB
                if idx >= len(metadata["filenames"]):
                    continue
                    
                filename = metadata["filenames"][idx]
                chunk_id = metadata["chunk_ids"][idx]
                source_kb = metadata["source_kbs"][idx]
                
                # Get actual chunk content from source KB
                chunk_text = get_chunk_from_source_kb(source_kb, chunk_id, filename)
                
            else:
                # Regular KB case
                if idx >= len(metadata["chunks"]):
                    continue
                
                chunk_text = metadata["chunks"][idx]
                filename = metadata["filenames"][idx] if idx < len(metadata["filenames"]) else "Unknown"
            
            # Filter by relevance threshold
            if relevance_threshold > 0.0 and score < relevance_threshold:
                print(f"[RAG_SETTINGS] Filtered out result with score {score:.4f} (below threshold {relevance_threshold})")
                continue
                
            # Safely handle chunk_text for preview - ensure it's a string
            if chunk_text is None:
                chunk_text = "Empty"
            elif not isinstance(chunk_text, str):
                chunk_text = str(chunk_text) if chunk_text else "Empty"
            
            semantic_results.append({
                "chunk": chunk_text,
                "filename": filename,
                "score": float(score),
                "semantic_score": float(score),
                "rank": i + 1
            })
        
        print(f"[RAG_SETTINGS] Initial semantic results: {len(semantic_results)} chunks")
        
        # Apply hybrid search if enabled (combine semantic with BM25 using RRF)
        if hybrid_search_enabled:
            print(f"[RAG_SETTINGS] Applying RRF hybrid search (bm25_weight ignored, using RRF fusion)")
            semantic_results = apply_hybrid_search(query, semantic_results, bm25_weight, metadata, is_universal_kb)
        
        # Apply reranking if enabled
        if reranking_enabled and len(semantic_results) > 1:
            print(f"[RAG_SETTINGS] Applying reranking with model: {reranking_model}")
            semantic_results = apply_reranking(query, semantic_results, reranking_model)
            # Limit to top_k_reranker
            semantic_results = semantic_results[:top_k_reranker]
            print(f"[RAG_SETTINGS] After reranking: {len(semantic_results)} chunks (limited to top_k_reranker: {top_k_reranker})")
        
        final_results = semantic_results[:top_k]
        print(f"[RAG_SETTINGS] Final results: {len(final_results)} chunks (limited to top_k: {top_k})")
        
        return final_results
        
    except Exception as e:
        st.error(f"Error searching vector database: {e}")
        return []

def search_vector_db(kb_name, query, top_k=None):
    """Search vector database with multi-query expansion for better retrieval"""
    try:
        # Check if query expansion is enabled in RAG settings
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'rag_settings'):
            rag_settings = st.session_state.rag_settings
            query_expansion_enabled = rag_settings.get("query_expansion", True)
            num_variations = rag_settings.get("num_query_variations", 3)
        else:
            query_expansion_enabled = True
            num_variations = 3
            
        # Use query expansion if enabled, has variations, and query is substantial
        if (query_expansion_enabled and num_variations > 0 and 
            query and len(query.strip()) > 10):
            print(f"[SEARCH_VDB] ✅ USING MULTI-QUERY RETRIEVAL ({num_variations + 1} total queries)")
            return expand_query_and_retrieve(kb_name, query, top_k or 5)
        else:
            reason = "expansion disabled" if not query_expansion_enabled else f"num_variations={num_variations}" if num_variations == 0 else "query too short"
            print(f"[SEARCH_VDB] ❌ USING SINGLE QUERY RETRIEVAL ({reason})")
            return search_vector_db_single(kb_name, query, top_k)
            
    except Exception as e:
        print(f"[SEARCH_VDB] Error in search_vector_db, falling back to single query: {e}")
        return search_vector_db_single(kb_name, query, top_k)

def apply_hybrid_search(query, semantic_results, bm25_weight, metadata, is_universal_kb):
    """Apply hybrid search using Reciprocal Rank Fusion (RRF) combining semantic and BM25 keyword search"""
    try:
        print(f"[RRF_DEBUG] Starting RRF hybrid search with {len(semantic_results)} documents")
        print(f"[RRF_DEBUG] Query: '{query}'")
        print(f"[RRF_DEBUG] Using RRF instead of linear combination (bm25_weight parameter ignored)")
        
        # Store original semantic ranking (already sorted by semantic similarity)
        semantic_ranking = [(i, result) for i, result in enumerate(semantic_results)]
        print(f"[RRF_DEBUG] Semantic ranking: {[f'Doc{i+1}:{result['score']:.4f}' for i, result in semantic_ranking[:5]]}")
        
        # Prepare documents for BM25
        documents = []
        doc_to_result_map = {}
        for i, result in enumerate(semantic_results):
            doc_tokens = result["chunk"].lower().split()
            documents.append(doc_tokens)
            doc_to_result_map[i] = result
            print(f"[RRF_DEBUG] Doc {i+1}: {len(doc_tokens)} tokens, semantic_score: {result.get('semantic_score', result['score']):.4f}")
        
        # Initialize BM25
        print(f"[RRF_DEBUG] Initializing BM25Okapi with {len(documents)} documents")
        bm25 = BM25Okapi(documents)
        
        # Tokenize query and get BM25 scores
        query_tokens = query.lower().split()
        print(f"[RRF_DEBUG] Query tokens: {query_tokens}")
        bm25_scores = bm25.get_scores(query_tokens)
        print(f"[RRF_DEBUG] Raw BM25 scores: {[f'{score:.4f}' for score in bm25_scores]}")
        
        # Create BM25 ranking (sort by BM25 score descending)
        bm25_ranking = [(i, score) for i, score in enumerate(bm25_scores)]
        bm25_ranking.sort(key=lambda x: x[1], reverse=True)
        print(f"[RRF_DEBUG] BM25 ranking: {[f'Doc{i+1}:{score:.4f}' for i, score in bm25_ranking[:5]]}")
        
        # Apply Reciprocal Rank Fusion (RRF)
        k = 60  # Standard RRF constant
        print(f"[RRF_DEBUG] Applying RRF with k={k}")
        
        rrf_scores = {}
        
        # Calculate RRF scores for each document
        for doc_idx in range(len(semantic_results)):
            # Find semantic rank (1-based)
            semantic_rank = doc_idx + 1  # semantic_results is already sorted by semantic score
            
            # Find BM25 rank (1-based)
            bm25_rank = None
            for rank, (idx, score) in enumerate(bm25_ranking):
                if idx == doc_idx:
                    bm25_rank = rank + 1
                    break
            
            if bm25_rank is None:
                bm25_rank = len(bm25_ranking) + 1  # Assign lowest rank if not found
            
            # Calculate RRF score: 1/(k + semantic_rank) + 1/(k + bm25_rank)
            rrf_score = 1.0 / (k + semantic_rank) + 1.0 / (k + bm25_rank)
            rrf_scores[doc_idx] = rrf_score
            
            # Store scores in result object
            result = semantic_results[doc_idx]
            result["semantic_rank"] = semantic_rank
            result["bm25_rank"] = bm25_rank
            result["bm25_score"] = bm25_scores[doc_idx]
            result["rrf_score"] = rrf_score
            result["score"] = rrf_score  # Update main score for ranking
            
            print(f"[RRF_DEBUG] Doc {doc_idx+1}: semantic_rank={semantic_rank}, bm25_rank={bm25_rank} -> RRF={rrf_score:.6f}")
        
        # Sort by RRF score (descending)
        semantic_results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        print(f"[RRF_DEBUG] Final RRF ranking: {[f'Doc:{r['rrf_score']:.6f}' for r in semantic_results[:5]]}")
        
        return semantic_results
        
    except Exception as e:
        print(f"[RRF_DEBUG] Error in RRF hybrid search: {e}")
        # Fallback to simplified BM25-like implementation if BM25Okapi fails
        return apply_hybrid_search_fallback(query, semantic_results, bm25_weight)

def apply_hybrid_search_fallback(query, semantic_results, bm25_weight):
    """Fallback BM25-like implementation if proper BM25 fails"""
    try:
        print(f"[BM25_FALLBACK_DEBUG] Using fallback BM25-like implementation")
        query_words = set(query.lower().split())
        print(f"[BM25_FALLBACK_DEBUG] Query words: {query_words}")
        
        for i, result in enumerate(semantic_results):
            chunk_text = result["chunk"].lower()
            chunk_words = set(chunk_text.split())
            
            # Simple BM25-like scoring: term frequency with normalization
            tf_score = 0
            for word in query_words:
                if word in chunk_text:
                    # Count occurrences and normalize by document length
                    word_count = chunk_text.count(word)
                    tf_score += word_count / len(chunk_words) if chunk_words else 0
            
            # Combine semantic score with BM25-like score
            semantic_score = result.get("semantic_score", result["score"])
            hybrid_score = (1 - bm25_weight) * semantic_score + bm25_weight * tf_score
            
            print(f"[BM25_FALLBACK_DEBUG] Doc {i+1}: semantic={semantic_score:.4f}, tf_score={tf_score:.4f} -> hybrid={hybrid_score:.4f}")
            
            result["bm25_score"] = tf_score
            result["hybrid_score"] = hybrid_score
            result["score"] = hybrid_score  # Update main score for ranking
        
        # Sort by hybrid score
        semantic_results.sort(key=lambda x: x.get("hybrid_score", x["score"]), reverse=True)
        print(f"[BM25_FALLBACK_DEBUG] Final ranking: {[f'{r.get('hybrid_score', r['score']):.4f}' for r in semantic_results]}")
        
        return semantic_results
        
    except Exception as e:
        print(f"[BM25_FALLBACK_DEBUG] Error in fallback hybrid search: {e}")
        return semantic_results

def apply_reranking(query, results, reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Apply reranking using actual MS-MARCO transformer models"""
    try:
        print(f"[RERANK_DEBUG] ===== Starting MS-MARCO Reranking =====")
        print(f"[RERANK_DEBUG] Query: '{query}'")
        print(f"[RERANK_DEBUG] Model: {reranking_model}")
        print(f"[RERANK_DEBUG] Input results: {len(results)} documents")
        
        for i, result in enumerate(results):
            original_score = result.get("semantic_score", result.get("hybrid_score", result["score"]))
            print(f"[RERANK_DEBUG] Input Doc {i+1}: score={original_score:.4f}, text='{result['chunk'][:60]}...'")
        
        # Load the reranking model
        print(f"[RERANK_DEBUG] Loading reranking model...")
        model = load_reranking_model(reranking_model)
        
        if model is None:
            print(f"[RERANK_DEBUG] Failed to load model {reranking_model}, falling back to custom scoring")
            return apply_reranking_fallback(query, results)
        
        print(f"[RERANK_DEBUG] Model loaded successfully: {type(model)}")
        
        # Prepare query-document pairs for the cross-encoder
        query_doc_pairs = []
        for i, result in enumerate(results):
            # Truncate documents to avoid token limits (most models have ~512 token limit)
            doc_text = result["chunk"][:2000]  # Rough character limit
            query_doc_pairs.append([query, doc_text])
            print(f"[RERANK_DEBUG] Prepared pair {i+1}: query_len={len(query)}, doc_len={len(doc_text)}")
        
        # Get reranking scores from the model
        print(f"[RERANK_DEBUG] Running inference on {len(query_doc_pairs)} query-document pairs...")
        rerank_scores = model.predict(query_doc_pairs)
        print(f"[RERANK_DEBUG] Raw reranking scores: {[f'{score:.4f}' for score in rerank_scores]}")
        
        # Update results with reranking scores
        print(f"[RERANK_DEBUG] Updating results with reranking scores...")
        for i, result in enumerate(results):
            original_score = result.get("semantic_score", result.get("hybrid_score", result["score"]))
            rerank_score = float(rerank_scores[i]) if i < len(rerank_scores) else 0
            
            print(f"[RERANK_DEBUG] Doc {i+1}: original={original_score:.4f} -> rerank={rerank_score:.4f}")
            
            # Store both original and rerank scores
            result["rerank_score"] = rerank_score
            result["original_score"] = original_score
            result["model_used"] = reranking_model
        
        # Sort by reranking scores (higher is better for cross-encoders)
        results.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        
        print(f"[RERANK_DEBUG] Final ranking order:")
        for i, result in enumerate(results):
            print(f"[RERANK_DEBUG]   {i+1}. Score: {result['rerank_score']:.4f} - '{result['chunk'][:60]}...'")
        
        print(f"[RERANK_DEBUG] Reranking completed. Top score: {results[0]['rerank_score']:.4f}")
        return results
        
    except Exception as e:
        print(f"[RERANK_DEBUG] Error in MS-MARCO reranking: {e}")
        return apply_reranking_fallback(query, results)

def apply_reranking_fallback(query, results):
    """Fallback reranking using local custom scoring if MS-MARCO models fail"""
    try:
        print("[RERANK_FALLBACK_DEBUG] ===== Using Fallback Reranking =====")
        print(f"[RERANK_FALLBACK_DEBUG] Using fallback reranking with local custom scoring")
        print(f"[RERANK_FALLBACK_DEBUG] Processing {len(results)} chunks")
        # Enhanced reranking with multiple scoring methods
        query_words = set(query.lower().split())
        
        for i, result in enumerate(results):
            chunk_text = result["chunk"].lower()
            chunk_words = set(chunk_text.split())
            
            # 1. Keyword overlap scoring
            keyword_overlap = len(query_words.intersection(chunk_words)) / len(query_words) if query_words else 0
            
            # 2. Query term density in chunk
            query_density = sum(chunk_text.count(word) for word in query_words) / len(chunk_text.split()) if chunk_text else 0
            
            # 3. Position-based scoring (earlier mentions get higher scores)
            position_score = 0
            for word in query_words:
                pos = chunk_text.find(word)
                if pos >= 0:
                    # Earlier positions get higher scores (inverse of normalized position)
                    position_score += (1 - pos / len(chunk_text)) if len(chunk_text) > 0 else 0
            position_score = position_score / len(query_words) if query_words else 0
            
            # 4. Exact phrase matching bonus
            phrase_bonus = 0.1 if query.lower() in chunk_text else 0
            
            # Combine all scoring methods
            original_score = result.get("semantic_score", result.get("hybrid_score", result["score"]))
            
            # Weight the different scoring components
            rerank_score = (
                0.4 * original_score +           # Original semantic/hybrid score
                0.3 * keyword_overlap +          # Keyword overlap
                0.2 * query_density +            # Query term density  
                0.1 * position_score +           # Position-based scoring
                phrase_bonus                     # Exact phrase bonus
            )
            
            result["rerank_score"] = rerank_score
            result["keyword_overlap"] = keyword_overlap
            result["query_density"] = query_density
            result["position_score"] = position_score
            result["phrase_bonus"] = phrase_bonus
            result["model_used"] = "custom_fallback"
        
        # Sort by reranked score
        results.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        
        print(f"[RERANK_FALLBACK_DEBUG] Completed reranking {len(results)} chunks")
        
        return results
        
    except Exception as e:
        print(f"[RERANK_FALLBACK_DEBUG] Error in fallback reranking: {e}")
        return results

def build_context_from_chunks(relevant_chunks):
    """Build context text from retrieved chunks using unified format"""
    if not relevant_chunks:
        return ""
    
    context_text = "\n\nRelevant context from uploaded documents:\n"
    for i, chunk in enumerate(relevant_chunks):
        score_display = f" (relevance: {chunk.get('rerank_score', chunk['score']):.3f})" if chunk.get('rerank_score') else f" (relevance: {chunk['score']:.3f})"
        context_text += f"\n[Source {i+1}: {chunk['filename']}{score_display}]\n{chunk['chunk']}\n"
    context_text += "\nPlease use this context to provide accurate and specific answers. Cite the source documents when referencing information from them."
    
    return context_text