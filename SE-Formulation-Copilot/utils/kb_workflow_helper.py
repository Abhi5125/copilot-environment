"""
Knowledge Base Workflow Helper
Utility functions for handling knowledge base selection and RAG processing in workflows.
"""

import streamlit as st
from utils.model_handler import get_workflow_model_client

# Import RAG functions
try:
    from utils.rag import search_vector_db, build_context_from_chunks
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


def get_workflow_kb_and_client(workflow_name, temperature=0.7):
    """
    Get the knowledge base selection and chat client for a workflow.
    
    Args:
        workflow_name: The workflow identifier (e.g., 'gate_product_developer', 'science_traceability_matrix')
        temperature: Temperature setting for the model
    
    Returns:
        tuple: (selected_kb_name, selected_kb, chat_client)
            - selected_kb_name: The name of the selected KB or "None"
            - selected_kb: The KB name for use with RAG (None if "None" selected)
            - chat_client: The configured chat client for this workflow
    """
    try:
        # Map workflow names to preference keys
        workflow_kb_mapping = {
            'main_chat': 'main_chat_kb',
            'science_traceability_matrix': 'stm_kb',
            'gate_product_developer': 'gate_product_kb',
            'proposal_assistant': 'main_chat_kb',  # Fallback to main chat KB
            'heritage_finder': 'main_chat_kb'      # Fallback to main chat KB
        }
        
        # Get the preference key for this workflow
        pref_key = workflow_kb_mapping.get(workflow_name, 'main_chat_kb')
        
        # Get KB preferences from user preferences
        kb_prefs = st.session_state.user_preferences.get("selected_knowledge_base", {})
        selected_kb_name = kb_prefs.get(pref_key, "None")
        
        # Get the actual KB object from the knowledge bases list
        selected_kb = None
        if selected_kb_name and selected_kb_name != "None":
            knowledge_bases = st.session_state.get("knowledge_bases", [])
            selected_kb = next((kb for kb in knowledge_bases if kb.get("name") == selected_kb_name), None)
        
        # Get the chat client using workflow model system
        chat_client = get_workflow_model_client(workflow_name, temperature=temperature, selected_kb=selected_kb)
        
        return selected_kb_name, selected_kb, chat_client
        
    except Exception as e:
        print(f"[KB_WORKFLOW] Error getting KB and client for {workflow_name}: {e}")
        # Return fallback values
        return "None", None, get_workflow_model_client(workflow_name, temperature=temperature)


def get_workflow_rag_context(user_message, selected_kb_name, chat_id_key=None, max_uploaded_chunks=3, max_kb_chunks=3):
    """
    Get RAG context from both uploaded files and knowledge base for a workflow.
    
    Args:
        user_message: The user's message to search for
        selected_kb_name: The selected knowledge base name ("None" if none selected)
        chat_id_key: Session state key for the chat ID (e.g., 'gate_chat_id', 'science_chat_id')
        max_uploaded_chunks: Maximum chunks to get from uploaded files
        max_kb_chunks: Maximum chunks to get from knowledge base
    
    Returns:
        list: Combined list of relevant chunks with metadata
    """
    relevant_chunks = []
    
    try:
        # Get chunks from uploaded files first (if chat_id_key provided)
        uploaded_chunks = []
        if chat_id_key and st.session_state.get(chat_id_key):
            uploaded_chunks = _get_uploaded_file_chunks(
                user_message, 
                st.session_state.get(chat_id_key),
                max_chunks=max_uploaded_chunks
            )
        
        # Get chunks from knowledge base
        kb_chunks = []
        if selected_kb_name and selected_kb_name != "None" and RAG_AVAILABLE:
            kb_chunks = _get_knowledge_base_chunks(
                user_message, 
                selected_kb_name,
                max_chunks=max_kb_chunks
            )
        
        # Combine and prioritize sources
        if uploaded_chunks:
            relevant_chunks = uploaded_chunks[:max_uploaded_chunks]
            if kb_chunks:
                # Add KB chunks but limit total
                remaining_slots = max(0, max_uploaded_chunks + max_kb_chunks - len(relevant_chunks))
                relevant_chunks.extend(kb_chunks[:remaining_slots])
        elif kb_chunks:
            relevant_chunks = kb_chunks[:max_kb_chunks + max_uploaded_chunks]  # Use full allocation if no uploads
        
        print(f"[KB_WORKFLOW] Retrieved {len(relevant_chunks)} total chunks ({len(uploaded_chunks)} uploaded, {len(kb_chunks)} KB)")
        return relevant_chunks
        
    except Exception as e:
        print(f"[KB_WORKFLOW] Error getting RAG context: {e}")
        return []


def prepare_rag_message(user_message, relevant_chunks):
    """
    Prepare the user message with RAG context if chunks are available.
    
    Args:
        user_message: The original user message
        relevant_chunks: List of relevant chunks from RAG
    
    Returns:
        str: The message with RAG context prepended, or original message if no chunks
    """
    if not relevant_chunks:
        return user_message
    
    try:
        context_text = "\n\n".join([chunk.get('content', '') for chunk in relevant_chunks])
        return f"Context from uploaded documents and knowledge base:\n{context_text}\n\nUser question: {user_message}"
    except Exception as e:
        print(f"[KB_WORKFLOW] Error preparing RAG message: {e}")
        return user_message


def _get_uploaded_file_chunks(user_message, chat_id, max_chunks=5):
    """
    Get relevant chunks from uploaded files for a specific chat.
    
    Args:
        user_message: The user's message
        chat_id: The chat ID for file embeddings
        max_chunks: Maximum number of chunks to return
    
    Returns:
        list: List of chunk dictionaries with content, filename, and score
    """
    try:
        # Import here to avoid circular imports
        from utils.model_handler import get_embeddings_client
        import numpy as np
        import faiss
        import pickle
        import os
        
        # Load chat embeddings
        db_path = f"databases/chat_embeddings/{chat_id}"
        
        if not (os.path.exists(f"{db_path}.faiss") and os.path.exists(f"{db_path}_metadata.pkl")):
            return []
        
        # Load FAISS index and metadata
        index = faiss.read_index(f"{db_path}.faiss")
        with open(f"{db_path}_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        if index.ntotal == 0:
            return []
        
        # Get embedding for the query
        embedding_client = get_embeddings_client()
        query_embedding = embedding_client.embed_documents([user_message])[0]
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search for relevant chunks
        scores, indices = index.search(query_vector, min(max_chunks, index.ntotal))
        
        # Format results
        chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(metadata["chunks"]):
                chunks.append({
                    'content': metadata["chunks"][idx],
                    'filename': metadata["filenames"][idx],
                    'score': float(score)
                })
        
        return chunks
        
    except Exception as e:
        print(f"[KB_WORKFLOW] Error retrieving uploaded file chunks: {e}")
        return []


def _get_knowledge_base_chunks(user_message, kb_name, max_chunks=3):
    """
    Get relevant chunks from a knowledge base.
    
    Args:
        user_message: The user's message
        kb_name: The knowledge base name  
        max_chunks: Maximum number of chunks to return
    
    Returns:
        list: List of chunk dictionaries
    """
    try:
        if not RAG_AVAILABLE:
            return []
        
        # Find the KB by name to get its ID
        from utils.data_persistence import load_custom_knowledge_bases
        kbs = load_custom_knowledge_bases()
        kb = next((kb for kb in kbs if kb.get("name") == kb_name), None)
        
        if kb:
            # Use KB ID if available, fallback to name
            kb_identifier = kb.get('id', kb_name)
            print(f"[KB_WORKFLOW] Using KB identifier: {kb_identifier} for KB: {kb_name}")
        else:
            # Fallback to name if KB not found
            kb_identifier = kb_name
            print(f"[KB_WORKFLOW] KB not found in list, using name: {kb_name}")
        
        from utils.rag import search_vector_db
        return search_vector_db(kb_identifier, user_message, top_k=max_chunks)
        
    except Exception as e:
        print(f"[KB_WORKFLOW] Error retrieving KB chunks from {kb_name}: {e}")
        return []


def store_temp_sources(workflow_name, relevant_chunks):
    """
    Store sources temporarily in session state for UI display.
    
    Args:
        workflow_name: The workflow identifier
        relevant_chunks: List of relevant chunks
    """
    if not relevant_chunks:
        return
    
    try:
        # Map workflow names to temp source keys
        temp_source_mapping = {
            'gate_product_developer': 'gate_temp_sources',
            'science_traceability_matrix': 'stm_temp_sources',
            'proposal_assistant': 'proposal_temp_sources',
            'heritage_finder': 'heritage_temp_sources',
            'main_chat': 'temp_sources'
        }
        
        temp_key = temp_source_mapping.get(workflow_name, 'temp_sources')
        st.session_state[temp_key] = relevant_chunks
        
    except Exception as e:
        print(f"[KB_WORKFLOW] Error storing temp sources for {workflow_name}: {e}")
