"""
Vector database management for Chat JPL application.
Handles FAISS vector databases for embeddings and search.
"""

import os
import pickle
import hashlib
import shutil
import streamlit as st
import numpy as np

# Try to import model handler dependencies with graceful fallback
try:
    from utils.model_handler import get_embeddings_client
    MODEL_HANDLER_AVAILABLE = True
except ImportError:
    MODEL_HANDLER_AVAILABLE = False
    get_embeddings_client = None

# Try to import FAISS with graceful fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.error("FAISS not available. Vector database functionality will be limited.")
    faiss = None

def get_vector_db_path(kb_identifier):
    """Get the file path for a knowledge base's vector database using KB ID only"""
    # Always expect KB ID (8 characters, alphanumeric)
    if not isinstance(kb_identifier, str) or len(kb_identifier) != 8 or not kb_identifier.replace('-', '').isalnum():
        raise ValueError(f"Invalid KB identifier '{kb_identifier}'. Expected 8-character alphanumeric ID.")
    
    base_path = f"databases/vector_dbs/kb_{kb_identifier}"
    return base_path

def get_kb_by_id(kb_id):
    """Get knowledge base data by ID"""
    try:
        # Import here to avoid circular imports
        from utils.data_persistence import load_custom_knowledge_bases
        kbs = load_custom_knowledge_bases()
        for kb in kbs:
            if kb.get('id') == kb_id:
                return kb
        return None
    except Exception as e:
        print(f"Error finding KB by ID: {e}")
        return None

def get_kb_by_name(kb_name):
    """Get knowledge base data by name"""
    try:
        # Import here to avoid circular imports
        from utils.data_persistence import load_custom_knowledge_bases
        kbs = load_custom_knowledge_bases()
        for kb in kbs:
            if kb.get('name') == kb_name:
                return kb
        return None
    except Exception as e:
        print(f"Error finding KB by name: {e}")
        return None

def load_vector_db(kb_identifier):
    """Load a FAISS index and its metadata for a KB"""
    try:
        # Get the DB path for this KB
        db_path = get_vector_db_path(kb_identifier)
        
        if not db_path:
            return None, None
        
        # Load FAISS index
        index_path = f"{db_path}.faiss"
        metadata_path = f"{db_path}_metadata.pkl"
        
        if not os.path.exists(index_path):
            return None, None
        
        index = faiss.read_index(index_path)
        
        # Load metadata
        metadata = []
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        
        return index, metadata
    except Exception as e:
        print(f"[VECTOR_DB] Error loading vector DB: {e}")
        return None, None

def save_vector_db(kb_identifier, index, metadata):
    """Save a FAISS index and its metadata"""
    try:
        # Get KB ID from name
        db_path = get_vector_db_path(kb_identifier)
        
        if not db_path:
            return False
        
        # Save index
        faiss.write_index(index, f"{db_path}.faiss")
        
        # Save metadata
        with open(f"{db_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        return True
    except Exception as e:
        print(f"[VECTOR_DB] Error saving vector DB: {e}")
        return False

def get_kb_id_from_name(kb_name):
    """Helper function to get KB ID from KB name"""
    try:
        from utils.data_persistence import load_custom_knowledge_bases
        kbs = load_custom_knowledge_bases()
        for kb in kbs:
            if kb.get('name') == kb_name:
                return kb.get('id')
        return None
    except Exception as e:
        print(f"Error getting KB ID from name: {e}")
        return None

def delete_file_from_knowledge_base(kb_name_or_id, filename_to_delete):
    """Delete a specific file from a knowledge base's vector database
    
    Args:
        kb_name_or_id: Either the KB name (e.g., 'KB 1') or KB ID (e.g., '8e612963')
        filename_to_delete: Name of the file to delete
    """
    print(f"[DELETE_FILE] Starting deletion of '{filename_to_delete}' from KB '{kb_name_or_id}'")
    
    if not FAISS_AVAILABLE:
        print(f"[DELETE_FILE] ERROR: Vector database not available")
        return False, "Vector database not available"
    
    # Convert KB name to ID if needed, or use the ID directly
    kb_identifier = get_kb_id_from_name(kb_name_or_id)
    
    # If get_kb_id_from_name returns None, assume kb_name_or_id is already an ID
    if not kb_identifier:
        print(f"[DELETE_FILE] No KB found with name '{kb_name_or_id}', assuming it's already a KB ID")
        kb_identifier = kb_name_or_id
    
    print(f"[DELETE_FILE] Loading vector database for KB identifier '{kb_identifier}'...")
    index, metadata = load_vector_db(kb_identifier)
    
    if index is None or metadata is None:
        print(f"[DELETE_FILE] ERROR: Knowledge base not found - index: {index}, metadata: {metadata}")
        return False, "Knowledge base not found"
    
    print(f"[DELETE_FILE] Loaded KB - index total: {index.ntotal}, metadata type: {type(metadata)}")
    
    # Check if metadata has the expected format
    if not isinstance(metadata, dict) or 'filenames' not in metadata:
        print(f"[DELETE_FILE] ERROR: Invalid metadata format - is dict: {isinstance(metadata, dict)}, has filenames: {'filenames' in metadata if isinstance(metadata, dict) else False}")
        return False, "Invalid metadata format"
    
    # Find indices of chunks from the file to delete
    indices_to_remove = []
    filenames = metadata.get('filenames', [])
    print(f"[DELETE_FILE] Total files in KB: {len(set(filenames))}, total chunks: {len(filenames)}")
    print(f"[DELETE_FILE] Unique files: {list(set(filenames))}")
    
    for i, filename in enumerate(filenames):
        if filename == filename_to_delete:
            indices_to_remove.append(i)
    
    if not indices_to_remove:
        print(f"[DELETE_FILE] ERROR: File '{filename_to_delete}' not found in knowledge base")
        print(f"[DELETE_FILE] Available files: {list(set(filenames))}")
        return False, f"File '{filename_to_delete}' not found in knowledge base"
    
    print(f"[DELETE_FILE] Found {len(indices_to_remove)} chunks to remove for file '{filename_to_delete}'")
    print(f"[DELETE_FILE] Chunk indices to remove: {indices_to_remove[:10]}{'...' if len(indices_to_remove) > 10 else ''}")
    
    # Create new metadata without the removed chunks
    print(f"[DELETE_FILE] Creating new metadata without removed chunks...")
    new_filenames = []
    new_chunks = []
    new_chunk_ids = []
    
    chunks = metadata.get('chunks', [])
    chunk_ids = metadata.get('chunk_ids', [])
    
    for i in range(len(filenames)):
        if i not in indices_to_remove:
            new_filenames.append(filenames[i])
            if i < len(chunks):
                new_chunks.append(chunks[i])
            if i < len(chunk_ids):
                new_chunk_ids.append(chunk_ids[i])
    
    print(f"[DELETE_FILE] New metadata - files: {len(set(new_filenames))}, chunks: {len(new_filenames)}")
    print(f"[DELETE_FILE] Remaining files: {list(set(new_filenames))}")
    
    # Create new index without the removed vectors
    print(f"[DELETE_FILE] Rebuilding vector index...")
    if new_filenames:  # If there are remaining chunks
        import numpy as np
        # Get all vectors except the ones to remove
        all_vectors = []
        print(f"[DELETE_FILE] Extracting {index.ntotal - len(indices_to_remove)} vectors from index...")
        for i in range(index.ntotal):
            if i not in indices_to_remove:
                vector = index.reconstruct(i)
                all_vectors.append(vector)
        
        if all_vectors:
            vectors_array = np.array(all_vectors)
            print(f"[DELETE_FILE] Vectors array shape: {vectors_array.shape}")
            
            # Create new index
            dimension = vectors_array.shape[1]
            new_index = faiss.IndexFlatIP(dimension)  # Use same index type as creation
            faiss.normalize_L2(vectors_array)  # Normalize for cosine similarity
            new_index.add(vectors_array)
            print(f"[DELETE_FILE] Created new index with {new_index.ntotal} vectors")
        else:
            # No vectors left, create empty index
            dimension = index.d
            new_index = faiss.IndexFlatIP(dimension)
            print(f"[DELETE_FILE] Created empty index (no vectors left)")
    else:
        # No chunks left, create empty index
        dimension = index.d
        new_index = faiss.IndexFlatIP(dimension)
        print(f"[DELETE_FILE] Created empty index (no chunks left)")
    
    # Update metadata
    new_metadata = {
        'filenames': new_filenames,
        'chunks': new_chunks,
        'chunk_ids': new_chunk_ids
    }
    
    print(f"[DELETE_FILE] Saving updated vector database...")
    print(f"[DELETE_FILE] Final stats - files: {len(set(new_filenames))}, chunks: {len(new_chunks)}, vectors: {new_index.ntotal}")
    
    # Save updated database
    if kb_identifier in ["Universal_Knowledge_Base"]:  # System KB - need special handling for save
        print(f"[DELETE_FILE] Warning: Cannot save system KB '{kb_name_or_id}' - manual regeneration required")
        return False, f"Cannot save system knowledge base '{kb_name_or_id}'"
    else:
        save_success = save_vector_db(kb_identifier, new_index, new_metadata)
    if save_success:
        print(f"[DELETE_FILE] ✅ Successfully deleted '{filename_to_delete}' ({len(indices_to_remove)} chunks)")
        print(f"[DELETE_FILE] ✅ Updated KB '{kb_name_or_id}' now has {len(set(new_filenames))} files and {len(new_chunks)} chunks")
        return True, f"Successfully deleted '{filename_to_delete}' ({len(indices_to_remove)} chunks)"
    else:
        print(f"[DELETE_FILE] ❌ Failed to save updated knowledge base")
        return False, "Failed to save updated knowledge base"

def add_documents_to_vector_db(kb_name, chunks, filename):
    """Add document chunks to a knowledge base's vector database"""
    if not MODEL_HANDLER_AVAILABLE or not FAISS_AVAILABLE:
        return False, 0
    
    # Convert KB name to ID for user KBs, or use name directly for system KBs
    system_kbs = ["Universal_Knowledge_Base"]
    kb_identifier = kb_name if kb_name in system_kbs else get_kb_id_from_name(kb_name)
    
    if not kb_identifier:
        print(f"[ADD_DOCS] ERROR: Could not find KB ID for '{kb_name}'")
        return False, 0
    
    try:
        # Get embedding client for this specific KB
        embedding_client = get_embeddings_client(kb_name)
        
        # Generate embeddings for chunks in batches of 8
        batch_size = 8
        embeddings = []
        print(f"[EMBEDDING] Starting embedding for {len(chunks)} chunks, batch size: {batch_size}")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            print(f"[EMBEDDING] Processing batch {i//batch_size+1} ({i} to {i+len(batch)-1})")
            try:
                # Extract text content from chunks for embedding
                batch_texts = []
                for chunk in batch:
                    if isinstance(chunk, dict) and 'text' in chunk:
                        batch_texts.append(chunk['text'])
                    elif isinstance(chunk, str):
                        batch_texts.append(chunk)
                    else:
                        print(f"[WARNING] Unknown chunk format in batch: {type(chunk)}")
                        batch_texts.append(str(chunk))
                
                print(f"[EMBEDDING] Processing embeddings for batch {i//batch_size+1} with {len(batch_texts)} texts...")
                
                # Add retry logic with timeout for embedding API call
                import time
                max_retries = 3
                retry_delay = 5
                
                for retry in range(max_retries):
                    try:
                        start_time = time.time()
                        batch_embeddings = embedding_client.embed_documents(batch_texts)
                        end_time = time.time()
                        
                        print(f"[EMBEDDING] Batch {i//batch_size+1} completed in {end_time - start_time:.2f}s, returned {len(batch_embeddings)} embeddings")
                        embeddings.extend(batch_embeddings)
                        break  # Success, exit retry loop
                        
                    except Exception as api_error:
                        print(f"[EMBEDDING] API ERROR in batch {i//batch_size+1}, attempt {retry + 1}/{max_retries}: {api_error}")
                        if retry == max_retries - 1:  # Last retry
                            print(f"[EMBEDDING] Failed after {max_retries} attempts, aborting upload")
                            return False, 0
                        else:
                            print(f"[EMBEDDING] Waiting {retry_delay}s before retry...")
                            time.sleep(retry_delay)
                
            except Exception as e:
                print(f"[EMBEDDING] ERROR in batch {i//batch_size+1}: {e}")
                return False, 0
        print(f"[EMBEDDING] Total embeddings collected: {len(embeddings)}")
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Load existing vector DB or create new one
        index, metadata = load_vector_db(kb_identifier)
        
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
        if save_vector_db(kb_identifier, index, metadata):
            return True, len(chunks)
        else:
            return False, 0
            
    except Exception as e:
        st.error(f"Error adding documents to vector database: {e}")
        return False, 0

# Chat-specific embedding functions
def get_chat_embedding_path(chat_id):
    """Get the file path for a chat's embeddings"""
    return f"databases/chat_embeddings/{chat_id}"

def create_chat_id(conversation_title):
    """Create a unique chat ID from conversation title with timestamp for uniqueness"""
    import time
    # Add timestamp to ensure uniqueness even for similar conversation titles
    unique_string = f"{conversation_title}_{int(time.time() * 1000000)}"  # microsecond precision
    return hashlib.md5(unique_string.encode()).hexdigest()[:8]

def load_chat_embeddings(chat_id):
    """Load chat-specific embeddings"""
    if not FAISS_AVAILABLE:
        return None, []
    
    base_path = get_chat_embedding_path(chat_id)
    faiss_path = f"{base_path}.faiss"
    metadata_path = f"{base_path}_metadata.pkl"
    
    if os.path.exists(faiss_path) and os.path.exists(metadata_path):
        try:
            index = faiss.read_index(faiss_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            return index, metadata
        except Exception as e:
            st.error(f"Error loading chat embeddings: {e}")
            return None, None
    return None, None

def save_chat_embeddings(chat_id, index, metadata):
    """Save chat-specific embeddings"""
    if not FAISS_AVAILABLE or index is None:
        return False
    
    base_path = get_chat_embedding_path(chat_id)
    faiss_path = f"{base_path}.faiss"
    metadata_path = f"{base_path}_metadata.pkl"
    
    # Create directory if it doesn't exist
    os.makedirs("databases/chat_embeddings", exist_ok=True)
    
    try:
        faiss.write_index(index, faiss_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        return True
    except Exception as e:
        st.error(f"Error saving chat embeddings: {e}")
        return False

def delete_chat_embeddings(chat_id):
    """Delete chat-specific embeddings"""
    base_path = get_chat_embedding_path(chat_id)
    faiss_path = f"{base_path}.faiss"
    metadata_path = f"{base_path}_metadata.pkl"
    
    try:
        if os.path.exists(faiss_path):
            os.remove(faiss_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        return True
    except Exception as e:
        st.error(f"Error deleting chat embeddings: {e}")
        return False

def process_uploaded_file(uploaded_file):
    """Process an uploaded file and return chunks"""
    content = ""
    
    if uploaded_file.type == "text/plain":
        content = str(uploaded_file.read(), "utf-8")
    elif uploaded_file.type == "application/pdf":
        import PyPDF2
        import io
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        for page in pdf_reader.pages:
            content += page.extract_text()
    else:
        st.error("Unsupported file type")
        return []
    
    # Simple chunking - split by paragraphs
    chunks = content.split('\n\n')
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    return chunks

def search_knowledge_base(query, kb_name, k=40):
    """Search knowledge base for relevant chunks using semantic search"""
    if not FAISS_AVAILABLE or not MODEL_HANDLER_AVAILABLE:
        print(f"[SEARCH_KB] FAISS or model handler not available")
        return []
    
    try:
        # Load vector database and metadata
        index, metadata = load_vector_db(kb_name)
        
        if index is None or not metadata.get("chunks"):
            return []
        
        # Get embedding client for this specific KB
        embedding_client = get_embeddings_client(kb_name)
        query_embedding = embedding_client.embed_documents([query])[0]
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Normalize query vector for cosine similarity
        import faiss
        faiss.normalize_L2(query_vector)
        
        # Search for similar chunks
        scores, indices = index.search(query_vector, min(k, index.ntotal))
        
        # Extract relevant chunks with metadata
        relevant_chunks = []
        chunks = metadata.get("chunks", [])
        filenames = metadata.get("filenames", [])
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(chunks):
                chunk_content = chunks[idx]
                
                # Ensure chunk content is a string (handle different chunk formats)
                if isinstance(chunk_content, dict):
                    if "text" in chunk_content:
                        chunk_content = chunk_content["text"]
                        print(f"[SEARCH_KB] Extracted text from dict chunk {idx}")
                    elif "page_content" in chunk_content:
                        chunk_content = str(chunk_content["page_content"])
                        print(f"[SEARCH_KB] Converted dict chunk {idx} to string (page_content)")
                    else:
                        chunk_content = str(chunk_content)
                        print(f"[SEARCH_KB] Converted dict chunk {idx} to string (full dict)")
                elif not isinstance(chunk_content, str):
                    chunk_content = str(chunk_content)
                    print(f"[SEARCH_KB] Converted {type(chunk_content)} chunk {idx} to string")
                
                relevant_chunks.append({
                    "content": chunk_content,
                    "filename": filenames[idx] if idx < len(filenames) else "Unknown",
                    "score": float(score)
                })
        
        print(f"[SEARCH_KB] Found {len(relevant_chunks)} relevant chunks for query in {kb_name}")
        return relevant_chunks
        
    except Exception as e:
        print(f"[SEARCH_KB] Error searching knowledge base {kb_name}: {e}")
        return []
