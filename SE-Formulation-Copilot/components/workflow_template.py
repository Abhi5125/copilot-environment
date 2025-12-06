"""
Generic Workflow Template for Chat JPL application.
This template provides a standardized structure for all workflow UIs.
"""

import streamlit as st
import time
import requests
import os
import numpy as np
import faiss
import pickle
from datetime import datetime
from utils.model_handler import get_chat_client, get_workflow_kb_and_client
from utils.data_persistence import save_conversations
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from components.top_nav import render_back_to_chat_button
from utils.vector_db import load_chat_embeddings, save_chat_embeddings, get_chat_embedding_path

# Import source rendering functions from reference handler
try:
    from utils.reference_handler import render_sources_ui, show_source_modal, show_all_sources_modal
except ImportError:
    # Fallback if import fails
    def render_sources_ui(sources, message_index):
        if sources:
            # Header with exact same styling as main chat
            st.markdown(
                '<div style="margin-top:-10rem;margin-bottom:0rem;'
                'padding:0rem;'
                'color:#666;font-size:0.9rem;">References from</div>',
                unsafe_allow_html=True
            )
            
            for i, source in enumerate(sources):
                st.markdown(f"**Source {i+1}:** *{source.get('filename', 'Unknown')}*")
                st.markdown(f"```\n{source.get('content', source)[:500]}{'...' if len(source.get('content', source)) > 500 else ''}\n```")
                if i < len(sources) - 1:
                    st.markdown("---")

    def show_source_modal(filename, chunks, chunk_index, from_all_sources=False, files_dict=None, message_index=None):
        st.write(f"Source: {filename}")
    
    def show_all_sources_modal(files_dict, message_index):
        st.write("All Sources")
        st.write("All Sources")

def save_workflow_conversation(workflow_name, messages_key, chat_id_key, title):
    """Save the current workflow conversation to the conversations list"""
    if st.session_state.get(chat_id_key) and st.session_state.get(messages_key):
        from datetime import datetime
        
        # Initialize conversations if not exists
        if 'conversations' not in st.session_state:
            st.session_state.conversations = []
        
        # Map workflow names to emoji and short prefixes
        workflow_prefixes = {
            'proposal_writing': '✍️ Proposal Writer:',
            'ao_comparison': '📋 AO Checker:',
            'science_traceability_matrix': '🔬 STM:',
            'heritage_finder': '🏛️ Heritage:',
            'gate_product_developer': '📋 GPD:'
        }
        
        # Check if conversation already exists
        existing_index = None
        conversation_title = st.session_state.get('current_conversation', title)
        # Ensure conversation_title is a string
        if conversation_title is None:
            conversation_title = title
        else:
            conversation_title = str(conversation_title)

        # Add workflow prefix to title if not already present
        workflow_prefix = workflow_prefixes.get(workflow_name, f'🔧 {workflow_name.upper()}:')
        if not conversation_title.startswith(workflow_prefix.split(':')[0]):
            conversation_title = f"{workflow_prefix} {conversation_title}"
        
        for i, conv in enumerate(st.session_state.conversations):
            if conv.get('chat_id') == st.session_state[chat_id_key]:
                existing_index = i
                break
        
        # Get uploaded files for this conversation
        chat_id = st.session_state.get(chat_id_key)
        uploaded_files = []
        if chat_id:
            for key in st.session_state.keys():
                # Look for keys that match the pattern: workflow_chatid_filename_size
                # Since chat_id already contains workflow_session_timestamp, we need to be more careful
                expected_prefix = f"{workflow_name}_{chat_id}_"
                if key.startswith(expected_prefix):
                    # Extract filename and size from everything after the expected prefix
                    remainder = key[len(expected_prefix):]
                    # Split by last underscore to separate filename from size
                    filename_parts = remainder.rsplit('_', 1)
                    if len(filename_parts) == 2:
                        filename = filename_parts[0]
                        try:
                            file_size = int(filename_parts[1])
                            uploaded_files.append({
                                'filename': filename,
                                'size': file_size,
                                'session_key': key
                            })
                        except ValueError:
                            # Skip if size is not a valid integer
                            continue
        
        conversation_data = {
            'title': conversation_title,
            'messages': st.session_state[messages_key],
            'date': datetime.now().strftime('%b %d, %I:%M %p'),
            'timestamp': datetime.now().isoformat(),
            'chat_id': st.session_state[chat_id_key],
            'type': workflow_name,
            'uploaded_files': uploaded_files  # Save file information
        }
        
        if existing_index is not None:
            # Remove existing conversation and add updated one at the end (most recent)
            st.session_state.conversations.pop(existing_index)
            st.session_state.conversations.append(conversation_data)
        else:
            st.session_state.conversations.append(conversation_data)
        
        save_conversations(st.session_state.conversations)

def restore_conversation_files(workflow_name, chat_id, uploaded_files_data):
    """Restore file session state when loading a conversation"""
    if not uploaded_files_data or not chat_id:
        return
    
    # Clear existing file states for this conversation first
    keys_to_remove = []
    expected_prefix = f"{workflow_name}_{chat_id}_"
    for key in st.session_state.keys():
        if key.startswith(expected_prefix):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    # Restore file states with correct session keys
    for file_data in uploaded_files_data:
        filename = file_data.get('filename')
        file_size = file_data.get('size')
        if filename and file_size:
            # Create the correct session key format: workflow_chatid_filename_size
            file_key = f"{workflow_name}_{chat_id}_{filename}_{file_size}"
            st.session_state[file_key] = True
            print(f"[RESTORE] Restored file: {filename} ({file_size} bytes) with key: {file_key}")

def process_uploaded_file_content(uploaded_file):
    """Process uploaded file by sending to chunking endpoint"""
    try:
        # Send file to chunking endpoint
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        chunking_url = st.session_state.get("chunking_url", "http://localhost:9876")
        
        file_size = getattr(uploaded_file, 'size', 0)
        print(f"[WORKFLOW_UPLOAD] Processing {uploaded_file.name} (size: {file_size} bytes)")
        
        response = requests.post(f"{chunking_url}/process", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"[WORKFLOW_UPLOAD] Response type: {type(result)}")
            
            # Handle different response formats
            if isinstance(result, list):
                # Check if it's a list of objects with page_content
                if result and isinstance(result[0], dict) and "page_content" in result[0]:
                    chunks = [item["page_content"] for item in result]
                    print(f"[WORKFLOW_UPLOAD] Extracted {len(chunks)} chunks from page_content format")
                else:
                    # If response is directly a list of chunks
                    chunks = result
                    print(f"[WORKFLOW_UPLOAD] Received direct list of {len(chunks)} chunks")
            elif isinstance(result, dict):
                # If response is a dictionary with chunks key
                chunks = result.get("chunks", [])
                print(f"[WORKFLOW_UPLOAD] Received dict with {len(chunks)} chunks")
            else:
                print(f"[WORKFLOW_UPLOAD] Unexpected response format: {type(result)}")
                chunks = []
            
            # Ensure all chunks are strings
            validated_chunks = []
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    # If chunk is a dict, try to extract page_content or convert to string
                    if "page_content" in chunk:
                        validated_chunks.append(str(chunk["page_content"]))
                    else:
                        validated_chunks.append(str(chunk))
                    print(f"[WORKFLOW_UPLOAD] Converted dict chunk {i} to string")
                elif isinstance(chunk, str):
                    validated_chunks.append(chunk)
                else:
                    validated_chunks.append(str(chunk))
                    print(f"[WORKFLOW_UPLOAD] Converted {type(chunk)} chunk {i} to string")
            
            chunks = validated_chunks
            print(f"[WORKFLOW_UPLOAD] Final validation: {len(chunks)} string chunks")
            
            print(f"[WORKFLOW_UPLOAD] Successfully processed {uploaded_file.name}: {len(chunks)} chunks")
            return chunks
        else:
            st.error(f"Error processing {uploaded_file.name}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def add_to_chat_embeddings(filename: str, chunks, chat_id: str):
    """Add document chunks to chat-specific embeddings"""
    try:
        from utils.vector_db import load_chat_embeddings, save_chat_embeddings
        from utils.model_handler import get_embeddings_client
        import numpy as np
        import faiss
        
        # Get embedding client
        embedding_client = get_embeddings_client()
        print(f"[EMBEDDING] Starting embedding generation for {len(chunks)} chunks")
        
        # Generate embeddings for chunks in batches
        batch_size = 8
        embeddings = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(chunks), batch_size):
            batch = chunks[batch_idx:batch_idx+batch_size]
            batch_num = (batch_idx // batch_size) + 1
            print(f"[EMBEDDING] Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            try:
                batch_embeddings = embedding_client.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                print(f"[EMBEDDING] Successfully generated embeddings for batch {batch_num}")
            except Exception as e:
                print(f"[EMBEDDING] Error generating embeddings for batch {batch_num}: {e}")
                return False, 0
        
        print(f"[EMBEDDING] Completed embedding generation for all {len(embeddings)} chunks")
        
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Load existing chat embeddings or create new
        index, metadata = load_chat_embeddings(chat_id)
        
        if index is None:
            # Create new FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dimension)
            metadata = {"chunks": [], "filenames": [], "chunk_ids": [], "source_types": []}
        else:
            # For existing index, check compatibility
            if index.d != embeddings_array.shape[1]:
                print(f"[EMBEDDING] Dimension mismatch: index={index.d}, chunk={embeddings_array.shape[1]}")
                return False, 0
        
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
            metadata["source_types"].append("uploaded_file")
        
        # Save updated embeddings
        save_chat_embeddings(chat_id, index, metadata)
        print(f"[EMBEDDING] Successfully processed {len(chunks)} chunks from {filename}")
        
        return True, len(chunks)
        
    except Exception as e:
        print(f"[EMBEDDING] Error adding documents to chat embeddings: {e}")
        return False, 0


def process_uploaded_files(uploaded_files, workflow_name, chat_id_key):
    """Process uploaded files and add them to workflow chat embeddings with progress tracking"""
    if not uploaded_files:
        return
    
    # Create progress indicators for multiple files
    if len(uploaded_files) > 1:
        st.markdown("### 📁 Processing Files")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Make file processing key chat-specific to isolate files per conversation
        chat_id = st.session_state.get(chat_id_key, 'default')
        file_key = f"{workflow_name}_{chat_id}_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
        if file_key not in st.session_state:
            # Update progress for multiple files
            if len(uploaded_files) > 1:
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            
            file_size = getattr(uploaded_file, 'size', 0)
            size_mb = file_size / (1024*1024) if file_size > 0 else 0
            
            # Show different spinner message for large files
            if size_mb > 100:
                spinner_msg = f"Processing large file {uploaded_file.name} ({size_mb:.1f} MB) - This may take several minutes..."
            elif size_mb > 50:
                spinner_msg = f"Processing {uploaded_file.name} ({size_mb:.1f} MB) - This may take a few minutes..."
            else:
                spinner_msg = f"Processing {uploaded_file.name}..."
            
            with st.spinner(spinner_msg):
                try:
                    # Send file to chunking endpoint for processing
                    chunks = process_uploaded_file_content(uploaded_file)
                    if chunks:
                        # Create chat ID if not exists
                        if st.session_state.get(chat_id_key) is None:
                            # Create temporary chat ID for workflow conversation
                            temp_title = f"{workflow_name}_chat_{int(time.time())}"
                            st.session_state[chat_id_key] = temp_title
                            print(f"[{workflow_name.upper()}_FILE_UPLOAD] Created chat_id: {st.session_state[chat_id_key]}")
                        else:
                            print(f"[{workflow_name.upper()}_FILE_UPLOAD] Using existing chat_id: {st.session_state[chat_id_key]}")
                        
                        # Add to chat embeddings
                        print(f"[{workflow_name.upper()}_FILE_UPLOAD] Adding embeddings for {uploaded_file.name}")
                        success, num_chunks = add_to_chat_embeddings(
                            uploaded_file.name,
                            chunks, 
                            st.session_state[chat_id_key]
                        )
                        
                        if success:
                            st.session_state[file_key] = True
                        else:
                            st.error(f"❌ Failed to upload {uploaded_file.name}")
                    else:
                        st.error(f"❌ Could not process {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    # Complete progress tracking for multiple files
    if len(uploaded_files) > 1:
        progress_bar.progress(1.0)
        status_text.text("✅ All files processed!")
        time.sleep(0.5)  # Reduced sleep time
    
    # Set a flag to indicate files were processed instead of immediate rerun
    st.session_state[f"{workflow_name}_files_processed"] = True

def load_workflow_prechunked_file(workflow_name, chat_id_key, file_type, display_name, chunks_filename):
    """Load pre-chunked files into workflow chat for querying and retrieval"""
    try:
        chunks_file_path = os.path.join(os.path.dirname(__file__), "..", chunks_filename)
        
        print(f"[WORKFLOW_PRECHUNK] Loading {file_type} chunks from {chunks_filename}")
        
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by the separator and filter out empty chunks
        raw_chunks = [chunk.strip() for chunk in content.split('#--------------------------------------#') if chunk.strip()]
        print(f"[WORKFLOW_PRECHUNK] Found {len(raw_chunks)} raw chunks")
        
        # Format chunks for storage similar to uploaded files
        formatted_chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_data = {
                'text': chunk_text,
                'chunk_id': i,
                'source': display_name
            }
            formatted_chunks.append(chunk_data)
        
        # Get chat ID for this workflow session
        chat_id = st.session_state.get(chat_id_key)
        if not chat_id:
            print(f"[WORKFLOW_PRECHUNK] No chat_id found for key {chat_id_key}")
            return False
        
        print(f"[WORKFLOW_PRECHUNK] Adding {len(formatted_chunks)} chunks to chat {chat_id}")
        
        # Add chunks to the chat embeddings 
        if add_to_workflow_chat_embeddings(formatted_chunks, display_name, chat_id):
            # Mark this file as processed in session state (simulate file upload)
            # Create a simulated file size for session state tracking
            total_text_length = sum(len(chunk['text']) for chunk in formatted_chunks)
            file_key = f"{workflow_name}_{chat_id}_{display_name}_{total_text_length}"
            st.session_state[file_key] = True
            
            print(f"[WORKFLOW_PRECHUNK] Successfully loaded {file_type} file with session key: {file_key}")
            return True
        else:
            print(f"[WORKFLOW_PRECHUNK] Failed to add chunks to workflow chat embeddings")
            return False
        
    except Exception as e:
        print(f"[WORKFLOW_PRECHUNK] Error loading {file_type} chunks: {str(e)}")
        return False

def add_to_workflow_chat_embeddings(chunks, filename, chat_id):
    """Add chunks to workflow chat embeddings for retrieval"""
    try:
        from utils.model_handler import get_embeddings_client
        import numpy as np
        import faiss
        
        print(f"[WORKFLOW_EMBED] Adding {len(chunks)} chunks from {filename} to chat {chat_id}")
        
        # Load existing chat embeddings or create new ones
        index, metadata = load_chat_embeddings(chat_id)
        
        if index is None:
            # Create new embeddings
            print(f"[WORKFLOW_EMBED] Creating new embeddings for chat {chat_id}")
            embedding_client = get_embeddings_client()
            
            # Get embeddings for all chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = embedding_client.embed_documents(chunk_texts)
            
            # Create FAISS index
            embedding_dim = len(embeddings[0])
            index = faiss.IndexFlatIP(embedding_dim)
            
            # Convert to numpy array and normalize
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            index.add(embeddings_array)
            
            # Create metadata
            metadata = {
                "chunks": chunks,
                "filenames": [filename],
                "source_types": ["pre_chunked_file"]
            }
        else:
            # Add to existing embeddings
            print(f"[WORKFLOW_EMBED] Adding to existing embeddings for chat {chat_id}")
            embedding_client = get_embeddings_client()
            
            # Get embeddings for new chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = embedding_client.embed_documents(chunk_texts)
            
            # Convert to numpy array and normalize
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            
            # Add to existing index
            index.add(embeddings_array)
            
            # Update metadata
            metadata["chunks"].extend(chunks)
            metadata["filenames"].append(filename)
            metadata["source_types"].append("pre_chunked_file")
        
        # Save updated embeddings
        success = save_chat_embeddings(chat_id, index, metadata)
        
        if success:
            print(f"[WORKFLOW_EMBED] Successfully saved embeddings for chat {chat_id}")
            return True
        else:
            print(f"[WORKFLOW_EMBED] Failed to save embeddings for chat {chat_id}")
            return False
            
    except Exception as e:
        print(f"[WORKFLOW_EMBED] Error adding chunks to embeddings: {str(e)}")
        return False

def retrieve_relevant_chunks(query, chat_id, k=40):
    """Retrieve relevant chunks from chat embeddings using semantic search"""
    try:
        from utils.vector_db import load_chat_embeddings
        from utils.model_handler import get_embeddings_client
        import numpy as np
        import faiss
        
        print(f"[CHAT_SEARCH] Loading embeddings for chat_id: {chat_id}")
        
        # Load chat embeddings
        index, metadata = load_chat_embeddings(chat_id)
        if index is None or metadata is None or not metadata.get("chunks"):
            print(f"[CHAT_SEARCH] No embeddings found for chat_id: {chat_id}")
            return []
        
        print(f"[CHAT_SEARCH] Found {len(metadata.get('chunks', []))} chunks in embeddings")
        
        # Get embedding for the query
        embedding_client = get_embeddings_client()
        query_embedding = embedding_client.embed_documents([query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Get RAG settings for reranking
        import streamlit as st
        rag_settings = st.session_state.get("rag_settings", {})
        reranking_enabled = rag_settings.get("reranking_engine", False)
        
        # Adjust k based on reranking settings
        if reranking_enabled:
            search_k = k * 2  # Get more candidates for reranking
            final_k = rag_settings.get("top_k_reranker", k)
        else:
            search_k = k
            final_k = k
        
        # Search for similar chunks
        scores, indices = index.search(query_embedding, min(search_k, index.ntotal))
        
        print(f"[CHAT_SEARCH] Search returned {len(scores[0])} results")
        
        # Prepare results
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(metadata["chunks"]):
                relevant_chunks.append({
                    "content": metadata["chunks"][idx],
                    "filename": metadata["filenames"][idx],
                    "score": float(score)
                })
        
        # Apply reranking if enabled
        if reranking_enabled and relevant_chunks:
            print(f"[CHAT_SEARCH] Applying reranking to {len(relevant_chunks)} chunks")
            relevant_chunks = apply_reranking_to_chunks(query, relevant_chunks, rag_settings)
            # Limit to final_k after reranking
            relevant_chunks = relevant_chunks[:final_k]
            print(f"[CHAT_SEARCH] After reranking, returning {len(relevant_chunks)} chunks")
        
        print(f"[CHAT_SEARCH] Returning {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
        
    except Exception as e:
        print(f"[CHAT_SEARCH] Error retrieving chunks: {e}")
        import traceback
        traceback.print_exc()
        return []

def apply_reranking_to_chunks(query, chunks, rag_settings):
    """Apply reranking algorithm to improve chunk relevance (workflow template version)"""
    try:
        # Simple keyword overlap scoring for reranking
        query_words = set(query.lower().split())
        
        for chunk in chunks:
            content_words = set(chunk["content"].lower().split())
            keyword_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
            
            # Combine original score with keyword overlap using bm25_weight
            bm25_weight = rag_settings.get("bm25_weight", 0.5)
            chunk["rerank_score"] = (1 - bm25_weight) * chunk["score"] + bm25_weight * keyword_overlap
            
        # Sort by reranked score (highest first)
        chunks.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        
        # Update score to use reranked score for display
        for chunk in chunks:
            chunk["original_score"] = chunk["score"]
            chunk["score"] = chunk.get("rerank_score", chunk["score"])
        
        return chunks
    except Exception as e:
        print(f"[CHAT_SEARCH] Error in reranking: {e}")
        return chunks


def retrieve_relevant_kb_chunks(query, kb_name, k=40):
    """Retrieve relevant chunks from knowledge base"""
    try:
        from utils.vector_db import search_knowledge_base
        return search_knowledge_base(query, kb_name, k)
    except Exception as e:
        print(f"Error retrieving KB chunks: {e}")
        return []

def workflow_ui(
    workflow_name: str,
    title: str,
    description: str,
    system_prompt: str,
    welcome_message: str,
    messages_key: str,
    chat_id_key: str,
    input_key: str,
    generating_key: str
):
    """
    Generic workflow UI function that can be used for any workflow.
    
    Args:
        workflow_name: Internal name for the workflow (e.g., "proposal_assistant")
        title: Display title (e.g., "Proposal Assistant")
        description: Description shown under title
        system_prompt: System prompt for the LLM
        welcome_message: Initial message shown to users
        messages_key: Session state key for messages (e.g., "proposal_messages")
        chat_id_key: Session state key for chat ID (e.g., "proposal_chat_id")
        input_key: Session state key for input field (e.g., "proposal_input_key")
        generating_key: Session state key for generating state (e.g., "generating_proposal_response")
        
    Note: Knowledge base is dynamically selected from user preferences via the nav bar dropdown.
    """
    
    # Special case: If this is the proposal writing assistant within proposal assistant workflow
    if workflow_name == "proposal_writing" and "proposal_mode" in st.session_state:
        # Custom back button that goes to proposal assistant main page
        if st.button("← Back", key="back_to_proposal_main_from_writing"):
            st.session_state.proposal_mode = "main"
            st.rerun()
    # Special case: If this is the AO comparison within proposal assistant workflow
    elif workflow_name == "ao_comparison" and "proposal_mode" in st.session_state:
        # Custom back button that goes to review selection page
        if st.button("← Back", key="back_to_review_from_ao"):
            st.session_state.proposal_mode = "review_selection"
            st.rerun()
    else:
        # Default back to chat button for all other workflows
        render_back_to_chat_button()
        
    st.markdown(
    """
    <div style="margin-top:0; padding-top:0">
        <hr style="margin-top:0; margin-bottom:0.25rem; border:1px solid #e5e7eb;" />
    </div>
    """,unsafe_allow_html=True)
    
    # Page header
    st.markdown(f"""
    <div style="text-align: center; padding: 0.1rem 0 0.1rem 0;">
        <h1>{title}</h1>
        <p style="color: #666; font-size: 1.1rem;">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chunking URL if not set
    if "chunking_url" not in st.session_state:
        st.session_state.chunking_url = "http://localhost:9876"
    
    # Initialize messages if not exists or if empty (but preserve loaded conversations)
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []
        # Add welcome message only if we're starting fresh (not loading a conversation)
        st.session_state[messages_key].append({
            "role": "assistant",
            "content": welcome_message
        })
    elif len(st.session_state[messages_key]) == 0:
        # Add welcome message if messages list is empty
        st.session_state[messages_key].append({
            "role": "assistant",
            "content": welcome_message
        })

    # Note: Knowledge base is dynamically selected from user preferences 
    # via get_workflow_kb_and_client() function, not hardcoded here

    # Custom CSS for chat message styling
    st.markdown("""
    <style>
    /* Style for user messages - light blue background */
    div[data-testid="chat-message-user"] {
        background-color: #dcf2ff !important;
        border-radius: 18px 18px 4px 18px !important;
    }
    
    /* Style for assistant messages - light gray background */
    div[data-testid="chat-message-assistant"] {
        background-color: #f7f8fc !important;
        border-radius: 18px 18px 18px 4px !important;
    }
    
    /* Adjust text color for better contrast */
    div[data-testid="chat-message-user"] .stMarkdown p,
    div[data-testid="chat-message-assistant"] .stMarkdown p {
        color: #1f2937 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display conversation
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state[messages_key]):
            if message["role"] == "user":
                # Use chat_message for user messages with custom styling
                with st.chat_message("user"):
                    # Handle both text and image messages
                    if isinstance(message["content"], dict) and 'text' in message["content"]:
                        # This is a message with text content
                        st.markdown(message["content"]["text"])
                    else:
                        # Regular text message
                        st.markdown(message["content"])
            else:
                # Use chat_message for assistant messages to enable proper markdown rendering
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                
                # Show sources if available with modern UI
                sources = message.get("sources")
                if sources:
                    render_sources_ui(sources, i, is_workflow=True)
                else:
                    # Add margin when no sources  
                    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)

    # Handle source modal displays
    for key in list(st.session_state.keys()):
        if key.startswith("show_source_") and isinstance(st.session_state[key], dict):
            # Extract source info and show modal
            source_info = st.session_state[key]
            show_source_modal(
                source_info["filename"], 
                source_info["chunks"], 
                source_info["file_index"],
                from_all_sources=source_info.get("from_all_sources", False),
                files_dict=source_info.get("files_dict"),
                message_index=source_info.get("message_index")
            )
            del st.session_state[key]
        elif key.startswith("show_all_sources_") and isinstance(st.session_state[key], dict):
            # Handle back button from individual source to "All Sources" dialog
            message_index = key.split("_")[-1]
            files_dict = st.session_state[key]
            show_all_sources_modal(files_dict, int(message_index))
            del st.session_state[key]

    # Add some bottom padding for the input section
    # Add some spacing after the last message
    st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)

    # Initialize chat ID if not exists - create unique ID per session
    if chat_id_key not in st.session_state:
        import time
        st.session_state[chat_id_key] = f"{workflow_name}_session_{int(time.time())}"
    
    # Files are now handled in the main upload/process section below

    # Check if last assistant message has no sources to add top margin
    add_top_margin = False
    if st.session_state[messages_key]:
        # Get the last assistant message
        for msg in reversed(st.session_state[messages_key]):
            if msg["role"] == "assistant":
                sources = msg.get("sources", [])
                if not sources:  # No sources available
                    add_top_margin = True
                break
    
    # Add top margin if last assistant message has no sources
 
    # Show processed files indicator above file uploader if any exist
    chat_id = st.session_state.get(chat_id_key, 'default')
    processed_files_info = []
    
    # Check for processed files and extract file info
    if chat_id and chat_id != 'default':
        for key in st.session_state.keys():
            # Look for keys that match the pattern: workflow_chatid_filename_size
            expected_prefix = f"{workflow_name}_{chat_id}_"
            if key.startswith(expected_prefix):
                # Extract filename and size from everything after the expected prefix
                remainder = key[len(expected_prefix):]
                # Split by last underscore to separate filename from size
                filename_parts = remainder.rsplit('_', 1)
                if len(filename_parts) == 2:
                    filename = filename_parts[0]
                    try:
                        file_size = int(filename_parts[1])
                        
                        # Format file size
                        if file_size >= 1024*1024:
                            size_str = f"{file_size/(1024*1024):.1f} MB"
                        elif file_size >= 1024:
                            size_str = f"{file_size/1024:.1f} KB"
                        else:
                            size_str = f"{file_size} B"
                        
                        # Avoid duplicates
                        file_info = (filename, size_str, key)
                        if file_info not in processed_files_info:
                            processed_files_info.append(file_info)
                    except ValueError:
                        # Skip if size is not a valid integer
                        continue
    
    # File processing section (removed file dropdown display for cleaner UI)

    # File uploader - single unified uploader for all file types
    uploaded_files = None
    try:
        # Make file uploader key chat-specific to isolate uploads per conversation
        chat_id = st.session_state.get(chat_id_key, 'default')
        
        # Use a stable key that doesn't change during form submissions
        file_upload_key = f"{workflow_name}_file_upload_{chat_id}"
        
        uploaded_files = st.file_uploader(
            "Upload files",
            type=['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'md', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key=file_upload_key,
            label_visibility="collapsed",
            help="📄 **Documents**: PDF, Word, Excel, PowerPoint, text files for context and analysis\n📷 **Images**: JPG, PNG for visual analysis with AI"
        )
    except Exception as e:
        st.warning("⚠️ File upload temporarily unavailable. You can still chat normally.")
        st.caption(f"Technical details: {str(e)}")
        uploaded_files = None
    
    # Process uploaded files immediately
    if uploaded_files:
        chat_id = st.session_state.get(chat_id_key, 'default')
        
        # Separate documents from images
        documents = []
        images = []
        
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type.lower() if uploaded_file.type else ""
            file_name = uploaded_file.name.lower()
            
            # Check if it's an image
            if (file_type.startswith('image/') or 
                any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])):
                images.append(uploaded_file)
            else:
                documents.append(uploaded_file)
        
        # Process documents (for RAG)
        if documents:
            unprocessed_files = []
            for uploaded_file in documents:
                file_key = f"{workflow_name}_{chat_id}_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
                if file_key not in st.session_state:
                    unprocessed_files.append(uploaded_file)
            
            # Only process if there are actually new files
            if unprocessed_files:
                process_uploaded_files(unprocessed_files, workflow_name, chat_id_key)
                
                # Mark files as processed
                for uploaded_file in unprocessed_files:
                    file_key = f"{workflow_name}_{chat_id}_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
                    st.session_state[file_key] = True
        
        # Process images (for vision analysis)
        if images:
            from utils.image_processing import process_uploaded_image, validate_image_file
            processed_images = []
            
            for image_file in images:
                if validate_image_file(image_file):
                    processed_image = process_uploaded_image(image_file)
                    if processed_image:
                        processed_images.append(processed_image)
                else:
                    st.warning(f"⚠️ Unsupported image format: {image_file.name}")
            
            if processed_images:
                # Store images in session state for vision processing
                uploaded_images_key = f"{workflow_name}_uploaded_images"
                st.session_state[uploaded_images_key] = processed_images
            else:
                # Clear any existing images
                uploaded_images_key = f"{workflow_name}_uploaded_images"
                if uploaded_images_key in st.session_state:
                    del st.session_state[uploaded_images_key]
        else:
            # Clear images if no images uploaded
            uploaded_images_key = f"{workflow_name}_uploaded_images"
            if uploaded_images_key in st.session_state:
                del st.session_state[uploaded_images_key]

    # Generate a unique key for the input field to reset it
    if input_key not in st.session_state:
        st.session_state[input_key] = 0

    # Chat input form - don't use clear_on_submit to avoid race conditions
    with st.form(f"{workflow_name}_chat_form", clear_on_submit=False):
        col_input, col_submit = st.columns([5, 1])
        
        with col_input:
            user_input = st.text_area(
                "Message", 
                placeholder=f"Ask about {title.lower()}...",
                label_visibility="collapsed",
                key=f"{workflow_name}_chat_input_field_{st.session_state[input_key]}"
            )
        with col_submit:
            st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Send", width='stretch')

        if submitted and user_input:
            print(f"[WORKFLOW_UI] Form submitted with message: {user_input[:50]}...")
            print(f"[WORKFLOW_UI] Current messages count: {len(st.session_state[messages_key])}")
            
            # Create conversation title on first message (after welcome message)
            if len(st.session_state[messages_key]) == 1:  # Only welcome message exists
                # Use just the user input as title, emoji prefix will be added when saving
                conversation_title = f"{user_input[:40]}{'...' if len(user_input) > 40 else ''}"
                st.session_state.current_conversation = conversation_title
                print(f"[WORKFLOW_UI] Set conversation title: {conversation_title}")
            
            # Add user message immediately (text only)
            user_message_content = user_input
            
            st.session_state[messages_key].append({"role": "user", "content": user_message_content})
            print(f"[WORKFLOW_UI] Added user message, new count: {len(st.session_state[messages_key])}")
            
            # Increment the key to reset the input field on next render
            st.session_state[input_key] += 1
            
            # Set generating state (no rerun needed - let natural flow handle it)
            st.session_state[generating_key] = True
            print(f"[WORKFLOW_UI] Set generating state to True")
            
            # Force a rerun to show the user message and clear the input
            st.rerun()
    
    # Check if we need to generate a response
    if st.session_state.get(generating_key, False):
        print(f"[WORKFLOW_UI] Generating response for {workflow_name}")
        print(f"[WORKFLOW_UI] Messages count during generation: {len(st.session_state[messages_key])}")
        with st.spinner("Thinking..."):
            try:
                # Get the last user message
                last_user_message_data = None
                last_user_text = None
                
                for msg in reversed(st.session_state[messages_key]):
                    if msg["role"] == "user":
                        last_user_message_data = msg["content"]
                        if isinstance(last_user_message_data, dict) and 'text' in last_user_message_data:
                            last_user_text = last_user_message_data['text']
                        else:
                            last_user_text = last_user_message_data
                        break
                
                print(f"[WORKFLOW_UI] Last user message: {last_user_text[:50] if last_user_text else 'None'}...")
                
                if last_user_text:
                    # Get workflow KB and client
                    kb_name, selected_kb, chat_client = get_workflow_kb_and_client(workflow_name)
                    
                    # Retrieve relevant context from uploaded files (use text for retrieval)
                    relevant_chunks = retrieve_relevant_chunks(
                        last_user_text, 
                        st.session_state[chat_id_key], 
                        k=20
                    )
                    
                    # Use KB for RAG if selected_kb is set (not None or "None")
                    kb_chunks = []
                    if selected_kb:
                        kb_chunks = retrieve_relevant_kb_chunks(
                            last_user_text,
                            selected_kb,  # Use the actual KB, not the display name
                            k=20
                        )
                    # Combine all relevant chunks
                    all_chunks = relevant_chunks + kb_chunks
                    
                    # Prepare user message with RAG context
                    if all_chunks:
                        # Safely extract content from chunks, handling different formats
                        chunk_contents = []
                        for chunk in all_chunks:
                            if isinstance(chunk, dict):
                                if "content" in chunk:
                                    chunk_contents.append(str(chunk["content"]))
                                elif "page_content" in chunk:
                                    chunk_contents.append(str(chunk["page_content"]))
                                else:
                                    # If it's a dict but no recognized content key, convert to string
                                    chunk_contents.append(str(chunk))
                            else:
                                # If it's not a dict, convert to string
                                chunk_contents.append(str(chunk))
                        
                        context_text = "\n\n".join(chunk_contents)
                        user_message_text = f"Context from uploaded documents and knowledge base:\n{context_text}\n\nUser question: {last_user_text}"
                    else:
                        user_message_text = last_user_text
                    
                    # Prepare messages for the LLM including conversation history
                    messages = [SystemMessage(content=system_prompt)]
                    
                    # Add conversation history (excluding the last user message which we'll add with context)
                    for msg in st.session_state[messages_key][:-1]:  # Exclude the last user message
                        if msg["role"] == "user":
                            # Handle both text messages in history
                            if isinstance(msg["content"], dict) and 'text' in msg["content"]:
                                # This is a structured message - just use the text
                                messages.append(HumanMessage(content=msg["content"]["text"]))
                            else:
                                messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            messages.append(AIMessage(content=msg["content"]))
                    
                    # Add the current user message with RAG context - check for images
                    uploaded_images_key = f"{workflow_name}_uploaded_images"
                    uploaded_images = st.session_state.get(uploaded_images_key, [])
                    
                    # Map workflow names to model preference keys (same as model_handler.py)
                    workflow_model_mapping = {
                        'main_chat': 'main_chat',
                        'science_traceability_matrix': 'science_traceability_matrix',
                        'gate_product_developer': 'gate_product_developer', 
                        'proposal_assistant': 'proposal_assistant',
                        'proposal_writing': 'proposal_assistant',  # Maps to proposal_assistant model
                        'ao_comparison': 'proposal_assistant',     # Maps to proposal_assistant model
                        'heritage_finder': 'heritage_finder'
                    }
                    
                    # Get the correct model preference key for this workflow
                    model_pref_key = workflow_model_mapping.get(workflow_name, workflow_name)
                    current_model_name = st.session_state.user_preferences.get("workflow_models", {}).get(model_pref_key, "")
                    
                    if uploaded_images:
                        from utils.image_processing import is_vision_capable_model, create_vision_message_content
                        
                        if is_vision_capable_model(current_model_name):
                            # Create vision message with images
                            vision_content = create_vision_message_content(user_message_text, uploaded_images)
                            # Pass as dict to preserve complex content structure for vision models
                            messages.append({"role": "user", "content": vision_content})
                        else:
                            # Model doesn't support vision, use text only with image info
                            image_info = f"\n\nNote: {len(uploaded_images)} image(s) uploaded but current model doesn't support vision analysis."
                            messages.append(HumanMessage(content=user_message_text + image_info))
                        
                        # Clear uploaded images after processing
                        del st.session_state[uploaded_images_key]
                    else:
                        # No images, use regular text processing
                        messages.append(HumanMessage(content=user_message_text))
                    
                    # Get response from LLM
                    response = chat_client.invoke(messages)
                    
                    # Clear generating state after successful response
                    st.session_state[generating_key] = False
                    
                    # Add assistant response with sources
                    assistant_message = {
                        "role": "assistant",
                        "content": response.content
                    }
                    
                    if all_chunks:
                        assistant_message["sources"] = all_chunks
                    
                    st.session_state[messages_key].append(assistant_message)
                    
                    # Save conversation to history
                    save_workflow_conversation(workflow_name, messages_key, chat_id_key, title)
                    
                    print(f"[WORKFLOW_UI] Generated response and saved conversation")
                    
                    # Rerun to show the new response
                    st.rerun()
                    
            except Exception as e:
                st.session_state[messages_key].append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error while processing your request: {str(e)}\n\nPlease try again or rephrase your question."
                })
                st.session_state[generating_key] = False
                
                # Save conversation to history even on error
                save_workflow_conversation(workflow_name, messages_key, chat_id_key, title)
                
                print(f"[WORKFLOW_UI] Handled error and saved conversation")
                
                # Rerun to show the error message
                st.rerun()

    # Add bottom margin after the entire workflow UI
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
