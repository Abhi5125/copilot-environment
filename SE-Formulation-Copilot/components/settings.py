"""
Settings page component for Chat JPL application.
Handles RAG configuration and knowledge base management.
"""

import streamlit as st
import os
import json
import pickle
import requests
from utils.data_persistence import save_custom_knowledge_bases, save_rag_settings, load_knowledge_bases, load_custom_knowledge_bases, save_user_profile, save_user_preferences, generate_kb_id
from components.top_nav import render_back_to_chat_button

# Try to import vector db functions with graceful fallback
try:
    from utils.vector_db import load_vector_db, get_vector_db_path, add_documents_to_vector_db, delete_file_from_knowledge_base
    VECTOR_DB_AVAILABLE = True
except ImportError as e:
    st.error(f"Vector database functions not available: {e}")
    VECTOR_DB_AVAILABLE = False
    # Create dummy functions to prevent errors
    def load_vector_db(kb_name): return None, {}
    def get_vector_db_path(kb_name): return f"databases/vector_dbs/{kb_name}"
    def add_documents_to_vector_db(kb_name, chunks, filename): return False, 0
    def delete_file_from_knowledge_base(kb_name, filename): return False, "Vector DB not available"

def get_kb_vector_db_path(kb):
    """Get vector DB path for a knowledge base, handling both ID-based and name-based paths"""
    # Try to use KB ID first (preferred for new knowledge bases)
    if 'id' in kb and kb['id'] and len(kb['id']) == 8:
        try:
            return get_vector_db_path(kb['id'])
        except ValueError:
            # If ID format is invalid, fall back to name-based path
            pass
    
    # Fallback to name-based path for legacy knowledge bases
    kb_name = kb['name']
    if kb_name.endswith('_Knowledge_Base'):
        # Remove the suffix to get the base name
        base_name = kb_name[:-15]  # Remove "_Knowledge_Base"
        return f"databases/vector_dbs/{base_name}_Knowledge_Base"
    else:
        return f"databases/vector_dbs/{kb_name}_Knowledge_Base"

def get_available_model_options():
    """Get list of available model options for selection"""
    options = []
    
    # Get all models from user preferences available_models list
    user_models = st.session_state.user_preferences.get("available_models", [])
    for model in user_models:
        model_name = model.get("name", "")
        if model_name:
            options.append(model_name)  # Exact model name as listed
    
    return options

def get_model_index(options, selected_model):
    """Get the index of the selected model in the options list"""
    try:
        return options.index(selected_model)
    except ValueError:
        return 0  # Default to first option if not found

def is_file_already_in_knowledge_base(kb_name, filename):
    """Check if a file with the given filename already exists in the knowledge base"""
    try:
        # Convert KB name to ID for user KBs
        from utils.vector_db import get_kb_id_from_name
        kb_identifier = get_kb_id_from_name(kb_name)
        
        if not kb_identifier:
            print(f"[DUPLICATE_CHECK] Warning: Could not find KB ID for '{kb_name}'")
            return False
        
        # Load existing knowledge base
        print(f"[DUPLICATE_CHECK] Loading KB '{kb_name}' (ID: {kb_identifier}) to check for file '{filename}'")
        index, metadata = load_vector_db(kb_identifier)
        
        print(f"[DUPLICATE_CHECK] Index: {index}, Metadata type: {type(metadata)}")
        if index is not None:
            print(f"[DUPLICATE_CHECK] Index total: {index.ntotal}")
        
        if metadata and metadata.get("filenames"):
            # Check if filename already exists in the metadata
            existing_files = metadata["filenames"]
            unique_files = list(set(existing_files))
            print(f"[DUPLICATE_CHECK] Found {len(existing_files)} chunks from {len(unique_files)} unique files: {unique_files}")
            
            # Check both original filename and converted filename
            # For .ppt files, also check for .pptx version
            # For .doc files, also check for .docx version
            # For .xls files, also check for .xlsx version
            filenames_to_check = [filename]
            
            # Add converted filename variations
            name_without_ext = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == '.ppt':
                filenames_to_check.append(f"{name_without_ext}.pptx")
            elif ext == '.doc':
                filenames_to_check.append(f"{name_without_ext}.docx")
            elif ext == '.xls':
                filenames_to_check.append(f"{name_without_ext}.xlsx")
            elif ext == '.pptx':
                filenames_to_check.append(f"{name_without_ext}.ppt")
            elif ext == '.docx':
                filenames_to_check.append(f"{name_without_ext}.doc")
            elif ext == '.xlsx':
                filenames_to_check.append(f"{name_without_ext}.xls")
            
            print(f"[DUPLICATE_CHECK] Checking filenames: {filenames_to_check}")
            
            # Check if any variation exists
            for check_filename in filenames_to_check:
                if check_filename in existing_files:
                    print(f"[DUPLICATE_CHECK] File '{check_filename}' (variation of '{filename}') already exists in KB '{kb_name}'")
                    return True
            
            print(f"[DUPLICATE_CHECK] File '{filename}' not found in KB '{kb_name}' (existing files: {len(set(existing_files))} unique)")
            return False
        else:
            print(f"[DUPLICATE_CHECK] No existing files found in KB '{kb_name}' - metadata: {metadata}")
            return False
            
    except Exception as e:
        print(f"[DUPLICATE_CHECK] Error checking for duplicate file in KB: {e}")
        # If we can't check, allow the upload to proceed
        return False

def is_file_already_in_knowledge_base_with_batch_context(kb_name, filename, current_batch_files):
    """Check if a file already exists in the knowledge base, considering current batch files"""
    try:
        # Convert KB name to ID for user KBs
        from utils.vector_db import get_kb_id_from_name
        kb_identifier = get_kb_id_from_name(kb_name)
        
        if not kb_identifier:
            print(f"[DUPLICATE_CHECK] Warning: Could not find KB ID for '{kb_name}'")
            return False
        
        # Load existing knowledge base
        print(f"[DUPLICATE_CHECK] Loading KB '{kb_name}' (ID: {kb_identifier}) to check for file '{filename}' (batch context)")
        index, metadata = load_vector_db(kb_identifier)
        
        if metadata and metadata.get("filenames"):
            existing_files = metadata["filenames"]
            unique_files = list(set(existing_files))
            print(f"[DUPLICATE_CHECK] Found {len(existing_files)} chunks from {len(unique_files)} unique files in KB")
            
            # Check both original filename and converted filename variations
            filenames_to_check = [filename]
            
            # Add converted filename variations
            name_without_ext = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == '.ppt':
                filenames_to_check.append(f"{name_without_ext}.pptx")
            elif ext == '.doc':
                filenames_to_check.append(f"{name_without_ext}.docx")
            elif ext == '.xls':
                filenames_to_check.append(f"{name_without_ext}.xlsx")
            elif ext == '.pptx':
                filenames_to_check.append(f"{name_without_ext}.ppt")
            elif ext == '.docx':
                filenames_to_check.append(f"{name_without_ext}.doc")
            elif ext == '.xlsx':
                filenames_to_check.append(f"{name_without_ext}.xls")
            
            print(f"[DUPLICATE_CHECK] Checking filenames: {filenames_to_check}")
            print(f"[DUPLICATE_CHECK] Current batch files: {current_batch_files}")
            
            # Check if any variation exists in KB (but not in current batch)
            for check_filename in filenames_to_check:
                if check_filename in existing_files:
                    # If the exact same file is in current batch, that's OK (different variations being uploaded together)
                    # But if the EXACT SAME filename already exists in KB, it's a true duplicate
                    if check_filename == filename:
                        # This exact file already exists in KB - it's a true duplicate regardless of batch
                        print(f"[DUPLICATE_CHECK] File '{filename}' already exists in KB '{kb_name}' - true duplicate")
                        return True
                    elif check_filename not in current_batch_files:
                        # Different variation exists in KB and not in current batch - it's a duplicate
                        print(f"[DUPLICATE_CHECK] File '{check_filename}' (variation of '{filename}') already exists in KB '{kb_name}' and is not in current batch")
                        return True
                    else:
                        # Different variation exists but is in current batch - allow both
                        print(f"[DUPLICATE_CHECK] File '{check_filename}' exists in KB but different variation '{filename}' is in current batch - allowing")
            
            print(f"[DUPLICATE_CHECK] File '{filename}' not found in KB '{kb_name}' (or is part of current batch)")
            return False
        else:
            print(f"[DUPLICATE_CHECK] No existing files found in KB '{kb_name}'")
            return False
            
    except Exception as e:
        print(f"[DUPLICATE_CHECK] Error checking for duplicate file in KB: {e}")
        return False

def render_settings_page():
    # Make this print statement below look better for settings loaded
    """Render the Settings page"""
    print("[Settings] Rendered UI")
    # Add back to chat button at the top
    render_back_to_chat_button()
    st.markdown(
    """
    <div style="margin-top:0; padding-top:0">
        <hr style="margin-top:0; margin-bottom:0.25rem; border:1px solid #e5e7eb;" />
    </div>
    """,unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center;">Settings</h1>', unsafe_allow_html=True)
    

    
    # Show warning if vector database is not available
    if not VECTOR_DB_AVAILABLE:
        st.warning("⚠️ Vector database functionality is not fully available. Some features may be limited.")
    
    # Initialize available models in user preferences if not exists
    if "available_models" not in st.session_state.user_preferences:
        st.session_state.user_preferences["available_models"] = []
    
    # Use only the models from user preferences (no hardcoded duplicates)
    available_models = st.session_state.user_preferences["available_models"]
    
    # Settings content
    st.markdown("## 🔍 Retrieval Settings")
    st.markdown("Configure how documents are retrieved and ranked for more accurate responses.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Full Context Mode
        if "full_context_mode_checkbox" not in st.session_state:
            st.session_state.full_context_mode_checkbox = st.session_state.rag_settings["full_context_mode"]
            
        full_context_mode = st.checkbox(
            "Full Context Mode",
            key="full_context_mode_checkbox"
        )
        st.session_state.rag_settings["full_context_mode"] = st.session_state.full_context_mode_checkbox
        
        # Hybrid Search
        if "hybrid_search_checkbox" not in st.session_state:
            st.session_state.hybrid_search_checkbox = st.session_state.rag_settings["hybrid_search"]
            
        hybrid_search = st.checkbox(
            "Hybrid Search",
            key="hybrid_search_checkbox"
        )
        st.session_state.rag_settings["hybrid_search"] = st.session_state.hybrid_search_checkbox
        
        # Query Expansion
        if "query_expansion_checkbox" not in st.session_state:
            st.session_state.query_expansion_checkbox = st.session_state.rag_settings.get("query_expansion", True)
            
        query_expansion = st.checkbox(
            "Query Expansion",
            key="query_expansion_checkbox",
            help="Generate semantic rephrasings of queries for better retrieval"
        )
        st.session_state.rag_settings["query_expansion"] = st.session_state.query_expansion_checkbox
        
        # Number of Query Variations (only show if query expansion is enabled)
        if st.session_state.query_expansion_checkbox:
            if "num_query_variations_input" not in st.session_state:
                # Ensure we have a valid default value (>= 1)
                current_value = st.session_state.rag_settings.get("num_query_variations", 3)
                st.session_state.num_query_variations_input = max(1, current_value)
                
            num_variations = st.number_input(
                "Number of Query Variations",
                min_value=1,
                max_value=8,
                key="num_query_variations_input",
                help="Total queries = original + this many variations (1-8 recommended for performance)"
            )
            st.session_state.rag_settings["num_query_variations"] = st.session_state.num_query_variations_input
        else:
            # Set to 0 when query expansion is disabled
            st.session_state.rag_settings["num_query_variations"] = 0
        
        # Reranking Engine
        if "reranking_engine_checkbox" not in st.session_state:
            st.session_state.reranking_engine_checkbox = st.session_state.rag_settings["reranking_engine"]
            
        reranking_engine = st.checkbox(
            "Reranking Engine",
            key="reranking_engine_checkbox"
        )
        st.session_state.rag_settings["reranking_engine"] = st.session_state.reranking_engine_checkbox
        
        # Reranking Model
        reranking_models = [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2"
        ]
        
        if "reranking_model_selectbox" not in st.session_state:
            current_model = st.session_state.rag_settings["reranking_model"]
            st.session_state.reranking_model_selectbox = reranking_models.index(current_model) if current_model in reranking_models else 0
            
        reranking_model_index = st.selectbox(
            "Reranking Model",
            options=range(len(reranking_models)),
            format_func=lambda x: reranking_models[x],
            key="reranking_model_selectbox"
        )
        st.session_state.rag_settings["reranking_model"] = reranking_models[st.session_state.reranking_model_selectbox]
    
    with col2:
        # Top K
        if "top_k_input" not in st.session_state:
            st.session_state.top_k_input = st.session_state.rag_settings["top_k"]
        
        top_k = st.number_input(
            "Top K",
            min_value=1,
            max_value=100,
            key="top_k_input"
        )
        st.session_state.rag_settings["top_k"] = st.session_state.top_k_input
        
        # Top K Reranker
        if "top_k_reranker_input" not in st.session_state:
            st.session_state.top_k_reranker_input = st.session_state.rag_settings["top_k_reranker"]
            
        top_k_reranker = st.number_input(
            "Top K Reranker",
            min_value=1,
            max_value=50,
            key="top_k_reranker_input"
        )
        st.session_state.rag_settings["top_k_reranker"] = st.session_state.top_k_reranker_input
        
        # Relevance Threshold
        if "relevance_threshold_input" not in st.session_state:
            st.session_state.relevance_threshold_input = st.session_state.rag_settings["relevance_threshold"]
            
        relevance_threshold = st.number_input(
            "Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="relevance_threshold_input"
        )
        st.session_state.rag_settings["relevance_threshold"] = st.session_state.relevance_threshold_input
    
    # Save RAG Settings Button
    if st.button("Save RAG Settings", width='stretch', key="save_rag"):
        try:
            # Save the current RAG settings from session state to databases folder
            save_rag_settings(st.session_state.rag_settings)
            st.success("✅ RAG settings saved successfully!")
        except Exception as e:
            st.error(f"❌ Error saving RAG settings: {e}")
            print(f"Error saving RAG settings: {e}")
    
    st.markdown("---")
    
    # Model Selection Section
    st.markdown("## 🤖 Default Model Selection")
    st.markdown("Configure default chat and embedding models for Main Chat and Proposal Assistant. Note: Main Chat can override this with knowledge base selections.")
    
    # Initialize workflow models in user preferences if not exists
    if "workflow_models" not in st.session_state.user_preferences:
        st.session_state.user_preferences["workflow_models"] = {
            "main_chat": "bedrock-claude-3.7-sonnet",
            "proposal_assistant": "gpt-oss:120b-64k"
        }
    
    # Initialize workflow embedding models if not exists
    if "workflow_embedding_models" not in st.session_state.user_preferences:
        st.session_state.user_preferences["workflow_embedding_models"] = {
            "main_chat": None,
            "proposal_assistant": None
        }
    
    # Get available chat and embedding models
    chat_model_options = []
    embedding_model_options = []
    
    if available_models:
        chat_model_options = [model["name"] for model in available_models if model.get("type", "chat") == "chat"]
        embedding_model_options = [model["name"] for model in available_models if model.get("type", "chat") == "embedding"]
    
    # Check if no models are configured
    if not chat_model_options:
        st.warning("⚠️ No chat models configured yet. Please add chat models in the 'External Services' section below first.")
        st.info("💡 Once you add models, you can select them for Main Chat and Proposal Assistant here.")
        
        # Show disabled dropdowns with placeholder text
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox(
                "Main Chat - Chat Model",
                options=["No chat models configured"],
                disabled=True,
                key="main_chat_model_select_disabled",
                help="Add a chat model below to enable this selection"
            )
        with col2:
            st.selectbox(
                "Proposal Assistant - Chat Model", 
                options=["No chat models configured"],
                disabled=True,
                key="proposal_model_select_disabled",
                help="Add a chat model below to enable this selection"
            )
    else:
        # Main Chat Models
        st.markdown("### Main Chat")
        col1, col2 = st.columns(2)
        
        with col1:
            # Main Chat - Chat Model
            current_main_chat = st.session_state.user_preferences["workflow_models"].get("main_chat", chat_model_options[0])
            if current_main_chat not in chat_model_options:
                current_main_chat = chat_model_options[0]
                
            main_chat_model = st.selectbox(
                "Chat Model",
                options=chat_model_options,
                index=get_model_index(chat_model_options, current_main_chat),
                key="main_chat_model_select",
                help="Default chat model for main chat (can be overridden by knowledge base selection)"
            )
            st.session_state.user_preferences["workflow_models"]["main_chat"] = main_chat_model
        
        with col2:
            # Main Chat - Embedding Model
            if embedding_model_options:
                current_main_chat_emb = st.session_state.user_preferences["workflow_embedding_models"].get("main_chat", embedding_model_options[0])
                if current_main_chat_emb and current_main_chat_emb not in embedding_model_options:
                    current_main_chat_emb = embedding_model_options[0]
                elif not current_main_chat_emb:
                    current_main_chat_emb = embedding_model_options[0]
                    
                main_chat_embedding = st.selectbox(
                    "Embedding Model",
                    options=embedding_model_options,
                    index=get_model_index(embedding_model_options, current_main_chat_emb),
                    key="main_chat_embedding_select",
                    help="Default embedding model for main chat document processing"
                )
                st.session_state.user_preferences["workflow_embedding_models"]["main_chat"] = main_chat_embedding
            else:
                st.selectbox(
                    "Embedding Model",
                    options=["No embedding models configured"],
                    disabled=True,
                    key="main_chat_embedding_select_disabled",
                    help="Add an embedding model to enable this selection"
                )
        
        # Proposal Assistant Models
        st.markdown("### Proposal Assistant")
        col3, col4 = st.columns(2)
        
        with col3:
            # Proposal Assistant - Chat Model
            current_proposal = st.session_state.user_preferences["workflow_models"].get("proposal_assistant", chat_model_options[0])
            if current_proposal not in chat_model_options:
                current_proposal = chat_model_options[0]
                
            proposal_model = st.selectbox(
                "Chat Model",
                options=chat_model_options,
                index=get_model_index(chat_model_options, current_proposal),
                key="proposal_model_select",
                help="Chat model for Proposal Assistant workflow"
            )
            st.session_state.user_preferences["workflow_models"]["proposal_assistant"] = proposal_model
        
        with col4:
            # Proposal Assistant - Embedding Model
            if embedding_model_options:
                current_proposal_emb = st.session_state.user_preferences["workflow_embedding_models"].get("proposal_assistant", embedding_model_options[0])
                if current_proposal_emb and current_proposal_emb not in embedding_model_options:
                    current_proposal_emb = embedding_model_options[0]
                elif not current_proposal_emb:
                    current_proposal_emb = embedding_model_options[0]
                    
                proposal_embedding = st.selectbox(
                    "Embedding Model",
                    options=embedding_model_options,
                    index=get_model_index(embedding_model_options, current_proposal_emb),
                    key="proposal_embedding_select",
                    help="Embedding model for Proposal Assistant document processing"
                )
                st.session_state.user_preferences["workflow_embedding_models"]["proposal_assistant"] = proposal_embedding
            else:
                st.selectbox(
                    "Embedding Model",
                    options=["No embedding models configured"],
                    disabled=True,
                    key="proposal_embedding_select_disabled",
                    help="Add an embedding model to enable this selection"
                )
    
    # Save Model Settings Button
    if st.button("Save Model Settings", width='stretch', key="save_models"):
        try:
            # Only save if we have chat models configured
            if chat_model_options:
                save_user_preferences(st.session_state.user_preferences)
                st.success("✅ Model settings saved successfully!")
            else:
                st.warning("⚠️ No models configured to save. Please add a model first.")
        except Exception as e:
            st.error(f"❌ Error saving model settings: {e}")
            print(f"Error saving model settings: {e}")
    
    st.markdown("---")
    
    # Knowledge Base Management Section - Full functionality from backup
    st.markdown("## 🧠 Knowledge Bases")
    
    # Ensure custom knowledge bases are loaded from disk if not in session state
    if "custom_knowledge_bases" not in st.session_state:
        from utils.data_persistence import load_custom_knowledge_bases
        loaded_kbs = load_custom_knowledge_bases()
        st.session_state.custom_knowledge_bases = loaded_kbs
        # Sync with dropdown session state
        st.session_state.knowledge_bases = loaded_kbs
        # CRITICAL: Also update user_preferences to prevent overwrites
        st.session_state.user_preferences["custom_knowledge_bases"] = loaded_kbs
    else:
        # Ensure user_preferences is also synced
        st.session_state.user_preferences["custom_knowledge_bases"] = st.session_state.custom_knowledge_bases
    
    # Also check what's in the main knowledge_bases
    main_kbs = st.session_state.get("knowledge_bases", [])
    
    # Display existing knowledge bases with full editing capabilities
    custom_knowledge_bases = st.session_state.get("custom_knowledge_bases", [])
    
    # Debug information
    if len(custom_knowledge_bases) == 0:
        st.info("📝 No custom knowledge bases found. Create one below to get started!")
    
    for i, kb in enumerate(custom_knowledge_bases):
        with st.expander(f"📚 {kb['name']}", expanded=False):
            # Top controls row with name and delete button
            col_name, col_delete = st.columns([3, 1])
            
            with col_name:
                # Allow editing knowledge base name
                updated_name = st.text_input(
                    "Knowledge Base Name",
                    value=kb.get("name", ""),
                    key=f"edit_name_{i}"
                )
            
            with col_delete:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with input
                if st.button("Delete Knowledge Base", key=f"delete_kb_{i}"):
                    # Confirmation using session state
                    if f"confirm_delete_{i}" not in st.session_state:
                        st.session_state[f"confirm_delete_{i}"] = True
                        st.warning(f"⚠️ Are you sure you want to delete '{kb['name']}'? This action cannot be undone!")
                        st.rerun()
                    else:
                        # Actually delete the knowledge base
                        if st.session_state.get("knowledge_base") == kb["name"]:
                            remaining_kbs = [k for j, k in enumerate(custom_knowledge_bases) if j != i]
                            if remaining_kbs:
                                st.session_state.knowledge_base = remaining_kbs[0]["name"]
                            else:
                                # No remaining custom knowledge bases, reset to default
                                st.session_state.knowledge_base = "None"
                        
                        # Delete vector database files using KB ID (preferred) or name (fallback)
                        kb_id = kb.get('id')
                        if kb_id:
                            # Use KB ID for file paths
                            db_path = get_vector_db_path(kb_id)
                        else:
                            # Fallback to name-based path
                            db_path = get_kb_vector_db_path(kb)
                        
                        try:
                            if os.path.exists(f"{db_path}.faiss"):
                                os.remove(f"{db_path}.faiss")
                            if os.path.exists(f"{db_path}_metadata.pkl"):
                                os.remove(f"{db_path}_metadata.pkl")
                            
                            # Also try to remove legacy name-based files if they exist
                            if kb_id:
                                legacy_db_path = f"databases/vector_dbs/{kb['name']}_Knowledge_Base"
                                if os.path.exists(f"{legacy_db_path}.faiss"):
                                    os.remove(f"{legacy_db_path}.faiss")
                                    print(f"[CLEANUP] Removed legacy file: {legacy_db_path}.faiss")
                                if os.path.exists(f"{legacy_db_path}_metadata.pkl"):
                                    os.remove(f"{legacy_db_path}_metadata.pkl")
                                    print(f"[CLEANUP] Removed legacy file: {legacy_db_path}_metadata.pkl")
                        except Exception as e:
                            st.error(f"Error deleting vector database files: {e}")
                        
                        # Remove from list
                        st.session_state.custom_knowledge_bases.pop(i)
                        
                        # Save to file using the proper function
                        try:
                            save_custom_knowledge_bases(st.session_state.custom_knowledge_bases)
                            # CRITICAL: Also update the main user_preferences session state to prevent overwrites
                            st.session_state.user_preferences["custom_knowledge_bases"] = st.session_state.custom_knowledge_bases
                            # Sync with dropdown session state
                            st.session_state.knowledge_bases = st.session_state.custom_knowledge_bases
                        except Exception as e:
                            st.error(f"Error saving knowledge bases: {e}")
                        
                        # Clean up confirmation state
                        if f"confirm_delete_{i}" in st.session_state:
                            del st.session_state[f"confirm_delete_{i}"]
                        
                        st.success(f"✅ Deleted knowledge base '{kb['name']}'!")
                        st.rerun()
            
            # Show confirmation warning if deletion was requested
            if f"confirm_delete_{i}" in st.session_state:
                st.error(f"⚠️ **Confirm Deletion**: Are you sure you want to delete '{kb['name']}'? All documents and settings will be permanently lost!")
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ Yes, Delete", key=f"confirm_yes_{i}", type="primary"):
                        # Perform the actual deletion (same logic as above)
                        if st.session_state.get("knowledge_base") == kb["name"]:
                            remaining_kbs = [k for j, k in enumerate(custom_knowledge_bases) if j != i]
                            if remaining_kbs:
                                st.session_state.knowledge_base = remaining_kbs[0]["name"]
                            else:
                                # No remaining custom knowledge bases, reset to default
                                st.session_state.knowledge_base = "None"
                        
                        # Delete vector database files
                        db_path = get_kb_vector_db_path(kb)
                        try:
                            if os.path.exists(f"{db_path}.faiss"):
                                os.remove(f"{db_path}.faiss")
                            if os.path.exists(f"{db_path}_metadata.pkl"):
                                os.remove(f"{db_path}_metadata.pkl")
                        except Exception as e:
                            st.error(f"Error deleting vector database files: {e}")
                        
                        # Remove from list
                        st.session_state.custom_knowledge_bases.pop(i)
                        
                        # Save to file using the proper function
                        try:
                            save_custom_knowledge_bases(st.session_state.custom_knowledge_bases)
                            # CRITICAL: Also update the main user_preferences session state to prevent overwrites
                            st.session_state.user_preferences["custom_knowledge_bases"] = st.session_state.custom_knowledge_bases
                            # Sync with dropdown session state
                            st.session_state.knowledge_bases = st.session_state.custom_knowledge_bases
                        except Exception as e:
                            st.error(f"Error saving knowledge bases: {e}")
                        
                        # Clean up confirmation state
                        del st.session_state[f"confirm_delete_{i}"]
                        
                        st.success(f"✅ Deleted knowledge base '{kb['name']}'!")
                        st.rerun()
                
                with col_cancel:
                    if st.button("❌ Cancel", key=f"confirm_no_{i}"):
                        del st.session_state[f"confirm_delete_{i}"]
                        st.rerun()
            
            # Display description and allow editing
            updated_description = st.text_area(
                "Description",
                value=kb.get("description", ""),
                height=80,
                key=f"edit_description_{i}"
            )

            # Chat model selection - only show chat type models
            chat_model_options = []
            embedding_model_options = []
            
            if available_models:
                chat_model_options = [model["name"] for model in available_models if model.get("type", "chat") == "chat"]
                embedding_model_options = [model["name"] for model in available_models if model.get("type", "chat") == "embedding"]
            
            # Chat Model Selection
            if chat_model_options:
                # Get current chat model, defaulting to first available if not found
                current_chat_model = kb.get("base_model", chat_model_options[0])
                if current_chat_model not in chat_model_options:
                    current_chat_model = chat_model_options[0]
                    
                updated_chat_model = st.selectbox(
                    "Chat Model",
                    options=chat_model_options,
                    index=chat_model_options.index(current_chat_model),
                    key=f"edit_chat_model_{i}",
                    help="The chat model to use for conversations with this knowledge base"
                )
            else:
                st.selectbox(
                    "Chat Model",
                    options=["No chat models configured"],
                    disabled=True,
                    key=f"edit_chat_model_{i}_disabled",
                    help="Add a chat model in the 'External Services' section to enable this selection"
                )
                updated_chat_model = kb.get("base_model", "No model")
            
            # Embedding Model Selection
            if embedding_model_options:
                # Get current embedding model, defaulting to first available if not found
                current_embedding_model = kb.get("embedding_model", embedding_model_options[0] if embedding_model_options else None)
                if current_embedding_model and current_embedding_model not in embedding_model_options:
                    current_embedding_model = embedding_model_options[0]
                elif not current_embedding_model and embedding_model_options:
                    current_embedding_model = embedding_model_options[0]
                    
                updated_embedding_model = st.selectbox(
                    "Embedding Model",
                    options=embedding_model_options,
                    index=embedding_model_options.index(current_embedding_model) if current_embedding_model else 0,
                    key=f"edit_embedding_model_{i}",
                    help="The embedding model to use for document vectorization and search"
                )
            else:
                st.selectbox(
                    "Embedding Model",
                    options=["No embedding models configured"],
                    disabled=True,
                    key=f"edit_embedding_model_{i}_disabled",
                    help="Add an embedding model in the 'External Services' section to enable this selection"
                )
                updated_embedding_model = kb.get("embedding_model", None)

            # System prompt editing
            updated_system_prompt = st.text_area(
                "System Prompt",
                value=kb.get("system_prompt", ""),
                height=300,
                key=f"edit_system_prompt_{i}"
            )

            # Display success message if it exists in session state
            success_key = f"kb_save_success_{i}"
            if success_key in st.session_state:
                st.success(st.session_state[success_key])
                # Clear the message after displaying it
                del st.session_state[success_key]

            # Save KB configuration button
            if st.button(f"Save Knowledge Base Configuration", key=f"save_kb_config_{i}"):
                # Validate name
                if not updated_name.strip():
                    st.error("❌ Knowledge base name cannot be empty!")
                elif updated_name != kb['name'] and any(other_kb['name'] == updated_name for j, other_kb in enumerate(custom_knowledge_bases) if j != i):
                    st.error(f"❌ A knowledge base named '{updated_name}' already exists!")
                else:
                    old_name = kb['name']
                    
                    # Update all fields in session state
                    st.session_state.custom_knowledge_bases[i]["name"] = updated_name
                    st.session_state.custom_knowledge_bases[i]["description"] = updated_description
                    st.session_state.custom_knowledge_bases[i]["base_model"] = updated_chat_model  # Keep base_model key for backward compatibility
                    st.session_state.custom_knowledge_bases[i]["embedding_model"] = updated_embedding_model
                    st.session_state.custom_knowledge_bases[i]["system_prompt"] = updated_system_prompt
                    
                    # Save configuration to disk using the proper function
                    try:
                        save_custom_knowledge_bases(st.session_state.custom_knowledge_bases)
                        # CRITICAL: Also update the main user_preferences session state to prevent overwrites
                        st.session_state.user_preferences["custom_knowledge_bases"] = st.session_state.custom_knowledge_bases
                        # Sync with dropdown session state
                        st.session_state.knowledge_bases = st.session_state.custom_knowledge_bases
                        # Store success message in session state to preserve across rerun
                        st.session_state[f"kb_save_success_{i}"] = f"✅ Saved configuration for '{updated_name}'!"
                        # Trigger a rerun to update the form with the saved values
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving knowledge base configuration: {e}")
                        print(f"Error saving knowledge bases: {e}")

                    # If name changed, handle vector database file renaming
                    # Only update the name in the configuration - vector DB files stay the same
                    # because they're now identified by the KB ID, not the name
                    if old_name != updated_name:
                        # Update current knowledge base selection if it was the renamed one
                        if st.session_state.get("knowledge_base") == old_name:
                            st.session_state.knowledge_base = updated_name
                        
                        st.success(f"✅ Renamed '{old_name}' to '{updated_name}'! Vector database files remain linked via KB ID.")
            
            # File upload section
            st.markdown("### 📁 Upload Documents")
            
            # Show current vector database status
            kb_id = kb.get('id', kb['name'])  # Use ID if available, fallback to name
            index, metadata = load_vector_db(kb_id)
            if index is not None and index.ntotal > 0:
                unique_files = set(metadata['filenames'])
                st.info(f"📊 Current database: {index.ntotal} chunks from {len(unique_files)} files")
                
                # Show files in database
                with st.expander("View uploaded files", expanded=False):
                    for filename in sorted(unique_files):
                        file_chunks = [i for i, f in enumerate(metadata['filenames']) if f == filename]
                        
                        # Create columns for file name and delete button
                        col_file, col_delete = st.columns([4, 1])
                        
                        with col_file:
                            st.write(f"• **{filename}** ({len(file_chunks)} chunks)")
                        
                        with col_delete:
                            # Create simpler unique key for delete button
                            import time
                            simple_key = f"delete_{kb['name'].replace(' ', '_')}_{filename.replace('.', '_')}_{i}"
                            
                            if st.button("✕", key=simple_key, width='stretch'):
                                # Perform deletion with debug info
                                print(f"[DELETE_BUTTON] ========================================")
                                print(f"[DELETE_BUTTON] DELETE REQUEST INITIATED")
                                print(f"[DELETE_BUTTON] Target file: '{filename}'")
                                print(f"[DELETE_BUTTON] Target KB: '{kb['name']}'")
                                print(f"[DELETE_BUTTON] KB files before deletion: {len(unique_files)} files")
                                print(f"[DELETE_BUTTON] VECTOR_DB_AVAILABLE: {VECTOR_DB_AVAILABLE}")
                                print(f"[DELETE_BUTTON] ========================================")
                                
                                if not VECTOR_DB_AVAILABLE:
                                    st.error("❌ Vector database functions not available. Please check your installation.")
                                    print(f"[DELETE_BUTTON] ❌ ABORT: Vector DB not available")
                                else:
                                    try:
                                        print(f"[DELETE_BUTTON] Calling delete_file_from_knowledge_base...")
                                        kb_id = kb.get('id', kb['name'])  # Use ID if available, fallback to name
                                        success, message = delete_file_from_knowledge_base(kb_id, filename)
                                        print(f"[DELETE_BUTTON] Delete function returned: success={success}, message='{message}'")
                                        
                                        if success:
                                            st.success(f"✅ {message}")
                                            print(f"[DELETE_BUTTON] ✅ SUCCESS: '{filename}' deleted from '{kb['name']}'")
                                            print(f"[DELETE_BUTTON] Triggering page refresh...")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ {message}")
                                            print(f"[DELETE_BUTTON] ❌ FAILED: {message}")
                                    except Exception as e:
                                        st.error(f"❌ Error during deletion: {str(e)}")
                                        print(f"[DELETE_BUTTON] ❌ EXCEPTION: {e}")
                                        import traceback
                                        traceback.print_exc()
            else:
                st.info("📊 No documents uploaded yet")
            
            # File uploader - now supports multiple files (simulates folder upload)
            st.markdown("**📁 Upload Files**")
            st.markdown("*Select multiple files to upload them all at once (like uploading a folder)*")
            
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                type=['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'md'],
                key=f"upload_{i}",
                accept_multiple_files=True
            )
            
            if uploaded_files:
                # Initialize recently uploaded files tracker if not exists
                if f"recently_uploaded_{kb['name']}" not in st.session_state:
                    st.session_state[f"recently_uploaded_{kb['name']}"] = set()
                
                # Check for duplicates first
                files_to_process = []
                duplicate_files = []
                current_batch_files = [f.name for f in uploaded_files]  # Track current batch
                recently_uploaded = st.session_state[f"recently_uploaded_{kb['name']}"]
                
                for uploaded_file in uploaded_files:
                    # Skip duplicate check if this file was just uploaded in the current session
                    if uploaded_file.name in recently_uploaded:
                        print(f"[DUPLICATE_CHECK] Skipping duplicate check for '{uploaded_file.name}' - was just uploaded in this session")
                        continue
                    elif is_file_already_in_knowledge_base_with_batch_context(kb['name'], uploaded_file.name, current_batch_files):
                        duplicate_files.append(uploaded_file.name)
                    else:
                        files_to_process.append(uploaded_file)
            else:
                # Clear recently uploaded files when uploader is empty
                if f"recently_uploaded_{kb['name']}" in st.session_state:
                    st.session_state[f"recently_uploaded_{kb['name']}"].clear()
                    print(f"[DUPLICATE_CHECK] Cleared recently uploaded files for KB '{kb['name']}'")
            
            if uploaded_files:
                # Show duplicate file warnings
                if duplicate_files:
                    if len(duplicate_files) == 1:
                        st.warning(f"⚠️ **{duplicate_files[0]}** already exists in this knowledge base.")
                    else:
                        st.warning(f"⚠️ **{len(duplicate_files)} files** already exist in this knowledge base and will be skipped:\n" + 
                                  "\n".join([f"• {filename}" for filename in duplicate_files]))
                
                # If no files to process, show message but continue rendering the rest of the page
                if not files_to_process:
                    if len(uploaded_files) == len(duplicate_files):
                        st.info("ℹ️ All selected files already exist in this knowledge base.")
                else:
                    # Process all new files
                    total_files = len(files_to_process)
                    if total_files > 0:                        
                        # Create a progress bar for the batch upload
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        successful_uploads = 0
                        failed_uploads = []
                        total_chunks_created = 0
                        
                        # Terminal progress header
                        print(f"\n[UPLOAD_START] Processing {total_files} files for knowledge base '{kb['name']}'")
                        print("=" * 80)
                        
                        for file_index, uploaded_file in enumerate(files_to_process):
                            # Update progress
                            progress = (file_index + 1) / total_files
                            progress_bar.progress(progress)
                            
                            # Enhanced status text with chunk tracking
                            status_text.text(f"Processing file {file_index + 1}/{total_files}: {uploaded_file.name} | Total chunks: {total_chunks_created}")
                            
                            # Terminal progress
                            print(f"\n[FILE_{file_index + 1:02d}/{total_files:02d}] Processing: {uploaded_file.name}")
                            
                            # Create a unique key for this file to prevent reprocessing
                            file_key = f"processed_{kb['name']}_{uploaded_file.name}_{uploaded_file.size}"
                            
                            if file_key not in st.session_state:
                                try:
                                    # Send file to chunking endpoint
                                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                                    
                                    chunking_url = st.session_state.get("chunking_url", "http://localhost:9876")
                                    
                                    file_size = getattr(uploaded_file, 'size', 0)
                                    print(f"[CHUNK_PROCESS] Processing {uploaded_file.name} (size: {file_size} bytes)")
                                    
                                    response = requests.post(
                                        f"{chunking_url}/process",
                                        files=files
                                    )
                                    
                                    if response.status_code == 200:
                                        try:
                                            result = response.json()
                                            # Handle both formats: direct list or dict with "chunks" key
                                            if isinstance(result, list):
                                                # Direct list of chunks from endpoint
                                                chunks = result
                                                print(f"[SUCCESS] Received direct list of {len(chunks)} chunks")
                                            elif isinstance(result, dict):
                                                # Dictionary with "chunks" key
                                                chunks = result.get("chunks", [])
                                                print(f"[SUCCESS] Received dict with {len(chunks)} chunks")
                                            else:
                                                print(f"[ERROR] Unexpected response format from chunking service: {type(result)} - {result}")
                                                chunks = []
                                        except ValueError as json_error:
                                            print(f"[ERROR] Failed to parse JSON response: {json_error}")
                                            print(f"[ERROR] Raw response: {response.text}")
                                            chunks = []
                                        
                                        # Print chunk creation progress to terminal
                                        print(f"[CHUNK_PROGRESS] File: {uploaded_file.name} | Created {len(chunks)} chunks")
                                        total_chunks_created += len(chunks)
                                        
                                        if chunks:
                                            # Convert endpoint format to vector DB format
                                            processed_chunks = []
                                            for chunk in chunks:
                                                if isinstance(chunk, dict) and 'page_content' in chunk:
                                                    # Endpoint format: {"page_content": "text", "metadata": {...}}
                                                    processed_chunks.append({
                                                        'text': chunk['page_content'],
                                                        'metadata': chunk.get('metadata', {})
                                                    })
                                                elif isinstance(chunk, dict) and 'text' in chunk:
                                                    # Already in correct format
                                                    processed_chunks.append(chunk)
                                                elif isinstance(chunk, str):
                                                    # Plain text chunk
                                                    processed_chunks.append({'text': chunk, 'metadata': {}})
                                                else:
                                                    print(f"[WARNING] Unknown chunk format: {type(chunk)}")
                                                    continue
                                            
                                            # Show detailed chunk progress in terminal
                                            print(f"[CHUNK_DETAILS] Processing {len(processed_chunks)} chunks for '{uploaded_file.name}':")
                                            for idx, chunk in enumerate(processed_chunks, 1):
                                                chunk_preview = chunk.get('text', '')[:50] + "..." if len(chunk.get('text', '')) > 50 else chunk.get('text', '')
                                                print(f"  [{idx:3d}/{len(processed_chunks)}] Chunk {idx}: {chunk_preview}")
                                            
                                            # Add chunks to vector database
                                            print(f"[VECTOR_DB] Adding {len(processed_chunks)} chunks to knowledge base '{kb['name']}'...")
                                            success, num_chunks = add_documents_to_vector_db(
                                                kb['name'],  # Pass KB name, not ID
                                                processed_chunks, 
                                                uploaded_file.name
                                            )
                                            
                                            if success:
                                                print(f"[VECTOR_DB] ✅ Successfully added {num_chunks} chunks to '{kb['name']}'")
                                                successful_uploads += 1
                                                # Mark as processed
                                                st.session_state[file_key] = True
                                                # Add to recently uploaded files to skip duplicate check on rerun
                                                st.session_state[f"recently_uploaded_{kb['name']}"].add(uploaded_file.name)
                                            else:
                                                print(f"[VECTOR_DB] ❌ Failed to add chunks to '{kb['name']}'")
                                                failed_uploads.append(f"{uploaded_file.name} (failed to add to database)")
                                        else:
                                            print(f"[CHUNK_PROGRESS] ❌ No chunks extracted from {uploaded_file.name}")
                                            failed_uploads.append(f"{uploaded_file.name} (no content extracted)")
                                    else:
                                        print(f"[ERROR] Chunking service returned status {response.status_code} for {uploaded_file.name}")
                                        print(f"[ERROR] Response: {response.text}")
                                        failed_uploads.append(f"{uploaded_file.name} (processing failed - status {response.status_code})")
                                        
                                except requests.exceptions.ConnectionError:
                                    print(f"[ERROR] Connection error processing {uploaded_file.name} - chunking service may be down")
                                    failed_uploads.append(f"{uploaded_file.name} (chunking service unavailable)")
                                    
                                except Exception as e:
                                    error_msg = str(e)
                                    print(f"[ERROR] Exception processing {uploaded_file.name}: {error_msg}")
                                    failed_uploads.append(f"{uploaded_file.name} (error: {error_msg})")
                            else:
                                # File already processed
                                successful_uploads += 1
                    
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Terminal progress summary
                        print("\n" + "=" * 80)
                        print(f"[UPLOAD_COMPLETE] Summary for knowledge base '{kb['name']}':")
                        print(f"  • Files processed: {successful_uploads}/{total_files}")
                        print(f"  • Total chunks created: {total_chunks_created}")
                        print(f"  • Failed uploads: {len(failed_uploads)}")
                        print("=" * 80 + "\n")
                        
                        # Show results
                        if successful_uploads > 0:
                            st.success(f"✅ Successfully uploaded {successful_uploads} files! Created {total_chunks_created} chunks total.")
                        
                        if failed_uploads:
                            st.error(f"❌ Failed to upload {len(failed_uploads)} files:")
                            for failure in failed_uploads:
                                st.error(f"• {failure}")
                        
                        # Rerun to refresh the file list
                        if successful_uploads > 0:
                            st.rerun()
            
            # Knowledge base management buttons - only Clear Documents button
            if index is not None and index.ntotal > 0:
                if st.button("Clear Documents", key=f"clear_{i}", width='stretch'):
                    # Clear vector database
                    db_path = get_kb_vector_db_path(kb)
                    try:
                        # Debug logging
                        print(f"[CLEAR_DOCS] Attempting to clear KB: '{kb['name']}'")
                        print(f"[CLEAR_DOCS] DB path: {db_path}")
                        print(f"[CLEAR_DOCS] FAISS file: {db_path}.faiss (exists: {os.path.exists(f'{db_path}.faiss')})")
                        print(f"[CLEAR_DOCS] Metadata file: {db_path}_metadata.pkl (exists: {os.path.exists(f'{db_path}_metadata.pkl')})")
                        
                        if os.path.exists(f"{db_path}.faiss"):
                            os.remove(f"{db_path}.faiss")
                            print(f"[CLEAR_DOCS] Removed FAISS file")
                        if os.path.exists(f"{db_path}_metadata.pkl"):
                            os.remove(f"{db_path}_metadata.pkl")
                            print(f"[CLEAR_DOCS] Removed metadata file")
                        
                        # Clear session state keys for processed files in this knowledge base
                        # This prevents the "file already exists" issue after clearing
                        keys_to_remove = []
                        for key in st.session_state.keys():
                            if key.startswith(f"processed_{kb['name']}_"):
                                keys_to_remove.append(key)
                        
                        for key in keys_to_remove:
                            del st.session_state[key]
                            print(f"[CLEAR_DOCS] Removed session state key: {key}")
                        
                        print(f"[CLEAR_DOCS] Successfully cleared KB '{kb['name']}'")
                        st.success(f"Cleared all documents from {kb['name']}")
                        st.rerun()
                    except Exception as e:
                        print(f"[CLEAR_DOCS] Error: {e}")
                        st.error(f"Error clearing documents: {e}")
            else:
                st.button("No Documents", disabled=True, key=f"no_docs_{i}", width='stretch')

    st.markdown("---")
    
    st.markdown("## ➕ Add New Knowledge Base")
    
    # Initialize form reset counter
    if "form_reset_counter" not in st.session_state:
        st.session_state.form_reset_counter = 0
    
    # Form to add new knowledge base - use counter as key to force reset
    with st.form(f"new_kb_form_{st.session_state.form_reset_counter}"):
        new_name = st.text_input(
            "Knowledge Base Name",
            placeholder="e.g., Mars 2020 Mission KB"
        )
        
        new_description = st.text_area(
            "Description",
            placeholder="Brief description of what this knowledge base contains...",
            height=100
        )

        submitted = st.form_submit_button("✨ Create Knowledge Base", width='stretch')

        if submitted and new_name and new_description:
            if not available_models:
                st.error("❌ Cannot create knowledge base: No models configured. Please add models first in the 'External Services' section.")
            else:
                # Check if we have both chat and embedding models
                chat_models = [m for m in available_models if m.get("type", "chat") == "chat"]
                embedding_models = [m for m in available_models if m.get("type", "chat") == "embedding"]
                
                if not chat_models:
                    st.error("❌ Cannot create knowledge base: No chat models configured. Please add a chat model first.")
                elif not embedding_models:
                    st.error("❌ Cannot create knowledge base: No embedding models configured. Please add an embedding model first.")
                else:
                    existing_names = [kb["name"] for kb in st.session_state.get("custom_knowledge_bases", [])]
                    if new_name not in existing_names:
                        if "custom_knowledge_bases" not in st.session_state:
                            st.session_state.custom_knowledge_bases = []
                        
                        # Use first available chat and embedding models as defaults
                        default_chat_model = chat_models[0]["name"]
                        default_embedding_model = embedding_models[0]["name"]
                        
                        st.session_state.custom_knowledge_bases.append({
                            "id": generate_kb_id(),  # Add unique ID
                            "name": new_name,
                            "description": new_description,
                            "base_model": default_chat_model,
                            "embedding_model": default_embedding_model,
                            "system_prompt": ""
                        })
                    
                        # Save to file using the dedicated function
                        try:
                            print(f"[SETTINGS] Saving new KB '{new_name}' to disk...")
                            save_custom_knowledge_bases(st.session_state.custom_knowledge_bases)
                            # CRITICAL: Also update the main user_preferences session state to prevent overwrites
                            st.session_state.user_preferences["custom_knowledge_bases"] = st.session_state.custom_knowledge_bases
                            # Sync with the dropdown session state
                            st.session_state.knowledge_bases = st.session_state.custom_knowledge_bases
                            print(f"[SETTINGS] Successfully saved and synced. Session now has {len(st.session_state.custom_knowledge_bases)} KBs")
                            # Increment counter to reset form on next render
                            st.session_state.form_reset_counter += 1
                            st.success(f"✅ Created '{new_name}' knowledge base!")
                            st.rerun()  # Refresh the page to show the new knowledge base
                        except Exception as e:
                            st.error(f"Error saving knowledge base: {e}")
                            print(f"Error saving custom knowledge bases: {e}")
                    else:
                        st.error("A knowledge base with this name already exists!")
        elif submitted:
            # Form was submitted but validation failed
            if not new_name:
                st.error("Please enter a knowledge base name!")
            if not new_description:
                st.error("Please enter a description!")
        
        # Show additional debugging info when form is submitted

    st.markdown("---")
    
    # Conversation Management Section
    st.markdown("## 💬 Conversation Management")
    st.markdown("This will permanently delete all saved conversations from all workflow tools (Main Chat and Proposal Assistant).")
    
    # Initialize confirmation state with a different name
    if 'show_clear_confirmation' not in st.session_state:
        st.session_state.show_clear_confirmation = False
    
    if not st.session_state.show_clear_confirmation:
        if st.button("Clear All Conversations", key="clear_all_conversations_btn", width='stretch'):
            st.session_state.show_clear_confirmation = True
            st.rerun()
    else:
        st.warning("⚠️ Are you sure? This cannot be undone!")
        col_confirm, col_cancel = st.columns(2)
        
        with col_confirm:
            if st.button("Yes, Delete", key="confirm_delete_conversations_btn", width='stretch'):
                try:
                    # Clear conversations from session state
                    st.session_state.conversations = []
                    
                    # Save empty conversations list to file
                    from utils.data_persistence import save_conversations
                    save_conversations([])
                    
                    # Clear current conversation state
                    if 'current_conversation' in st.session_state:
                        del st.session_state.current_conversation
                    
                    # Reset confirmation state
                    st.session_state.show_clear_confirmation = False
                    
                    st.success("✅ All conversations have been cleared!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error clearing conversations: {str(e)}")
                    st.session_state.show_clear_confirmation = False
        
        with col_cancel:
            if st.button("Cancel", key="cancel_clear_conversations", width='stretch'):
                st.session_state.show_clear_confirmation = False
                st.rerun()

    # Show current conversation count
    conversation_count = len(st.session_state.get('conversations', []))
    if conversation_count > 0:
        st.info(f"📊 Currently storing {conversation_count} conversation(s)")
    else:
        st.info("📊 No conversations currently saved")

    st.markdown("---")
    st.markdown("## 🔗 External Services")
    
    # Display current available models
    st.markdown("#### Available Models")
    if available_models:
        # Add padding above headers
        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
        
        # Table header with Type column added
        col_model, col_type, col_test, col_delete = st.columns([5, 2, 2, 1], gap="small")
        
        with col_model:
            st.markdown("**Model**")
        with col_type:
            st.markdown("<div style='text-align: center;'><strong>Type</strong></div>", unsafe_allow_html=True)
        with col_test:
            st.markdown("<div style='text-align: center;'><strong>Test</strong></div>", unsafe_allow_html=True)
        with col_delete:
            st.markdown("<div style='text-align: center;'></div>", unsafe_allow_html=True)
        
        # Reduced spacing divider
        st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem; border: 1px solid #dee2e6;'>", unsafe_allow_html=True)
        
        # Table rows with consistent spacing and reduced gap
        for i, model in enumerate(available_models):
            col_model, col_type, col_test, col_delete = st.columns([5, 2, 2, 1], gap="small")
            
            with col_model:
                # Display model name and provider if available
                provider_info = model.get("provider", "Custom")
                
                model_display = model['name']
                if provider_info and provider_info != "Custom":
                    model_display += f" ({provider_info})"
                
                st.text(model_display)
            
            with col_type:
                # Display model type (Chat or Embedding)
                model_type = model.get("type", "chat")  # Default to chat for backward compatibility
                type_display = "💬 Chat" if model_type == "chat" else "🔤 Embedding"
                st.markdown(f"<div style='text-align: center;'>{type_display}</div>", unsafe_allow_html=True)
            
            with col_test:
                # Test button for all models (chat and embedding)
                if st.button("Test Connection", key=f"test_model_{i}", width='stretch'):
                    model_type = model.get("type", "chat")
                    
                    try:
                        headers = {
                            "Authorization": f"Bearer {model.get('api_key', '')}",
                            "Content-Type": "application/json"
                        }
                        
                        if model_type == "chat":
                            # Test chat model with a simple message
                            payload = {
                                "model": model["name"],
                                "messages": [{"role": "user", "content": "Hello, test message."}]
                            }
                        else:
                            # Test embedding model with a simple text
                            payload = {
                                "model": model["name"],
                                "input": "Test embedding connection"
                            }
                        
                        # Disable SSL verification for internal endpoints (like ChatHPC) with self-signed certificates
                        response = requests.post(model["endpoint"], headers=headers, json=payload, timeout=10, verify=False)
                        
                        if response.status_code == 200:
                            # Additional validation for embedding models
                            if model_type == "embedding":
                                response_json = response.json()
                                if "data" in response_json and len(response_json["data"]) > 0:
                                    embedding_dim = len(response_json["data"][0].get("embedding", []))
                                    st.success(f"✅ {model['name']}: Connection successful!")
                                else:
                                    st.success(f"✅ {model['name']}: Connection successful!")
                            else:
                                st.success(f"✅ {model['name']}: Connection successful!")
                        else:
                            st.error(f"❌ {model['name']}: HTTP {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ {model['name']}: {str(e)}")
            
            with col_delete:
                # All models can be deleted now
                if st.button("✕", key=f"delete_model_{i}", width='stretch'):
                    st.session_state.user_preferences["available_models"].pop(i)
                    save_user_preferences(st.session_state.user_preferences)
                    st.rerun()
    else:
        st.info("No custom models configured yet.")
    
    # Custom endpoint configuration to add new chat models
    st.markdown("#### Add Chat Model")
    st.markdown("Configure settings for a custom LLM and add it to available models")
    
    # Form to add new custom model
    with st.form("add_custom_model_form"):
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            new_model_name = st.text_input(
                "Model Name",
                placeholder="e.g., gpt-4, claude-3-sonnet, gpt-oss:120b-64k",
                help="The exact model name to use in API calls"
            )
            
            custom_endpoint_url = st.text_input(
                "Endpoint URL",
                placeholder="https://api.openai.com/v1/chat/completions",
                help="Full API endpoint URL for this model"
            )
        
        with col_form2:
            api_key = st.text_input(
                "API Key",
                placeholder="sk-... (OpenAI) or your-api-key",
                help="API key for authentication with this endpoint",
                type="password"
            )

        submitted = st.form_submit_button("Add Custom Model", width='stretch')

        if submitted:
            # Simple validation
            validation_error = None
            
            if not new_model_name:
                validation_error = "Model Name is required!"
            elif not api_key:
                validation_error = "API Key is required!"
            elif not custom_endpoint_url:
                validation_error = "Endpoint URL is required!"
            
            if validation_error:
                st.error(f"❌ {validation_error}")
            else:
                # Check if model already exists
                existing_names = [model["name"] for model in available_models]
                if new_model_name in existing_names:
                    st.error(f"❌ Model '{new_model_name}' already exists!")
                else:
                    # Add new custom chat model
                    new_model = {
                        "name": new_model_name,
                        "endpoint": custom_endpoint_url,
                        "api_key": api_key,
                        "type": "chat"
                    }
                    
                    st.session_state.user_preferences["available_models"].append(new_model)
                    save_user_preferences(st.session_state.user_preferences)
                    
                    # Set success flag and clear form flag
                    st.session_state.model_added_success = new_model_name
                    st.session_state.clear_model_form = True
                    st.rerun()
    
    # Display success message if a model was just added
    if st.session_state.get("model_added_success"):
        st.success(f"✅ Added custom model '{st.session_state.model_added_success}'!")
        # Clear the success flag after displaying
        del st.session_state.model_added_success
    
    st.markdown("---")
    
    # Embedding Model Configuration Section
    st.markdown("#### Add Embedding Model")
    st.markdown("Add a standalone embedding model for vector operations")
    
    # Use a form key that includes a counter to force form reset after successful submission
    form_key = "add_embedding_model_form"
    if st.session_state.get("clear_embedding_form"):
        # Increment counter to create new form instance
        counter = st.session_state.get("embedding_form_counter", 0) + 1
        st.session_state.embedding_form_counter = counter
        form_key = f"add_embedding_model_form_{counter}"
        del st.session_state.clear_embedding_form
    
    with st.form(form_key):
        col_emb1, col_emb2 = st.columns(2)
        
        with col_emb1:
            embedding_model_name = st.text_input(
                "Embedding Model Name",
                placeholder="e.g., text-embedding-3-large, text-embedding-ada-002",
                help="The name of the embedding model to use for vector embeddings",
                key=f"emb_name_{st.session_state.get('embedding_form_counter', 0)}"
            )
            
            embedding_endpoint_url = st.text_input(
                "Endpoint URL",
                placeholder="https://api.openai.com/v1/embeddings",
                help="Full API endpoint URL for this embedding model",
                key=f"emb_endpoint_{st.session_state.get('embedding_form_counter', 0)}"
            )
        
        with col_emb2:
            embedding_api_key = st.text_input(
                "API Key",
                placeholder="sk-... (OpenAI) or your-api-key",
                help="API key for authentication with this endpoint",
                type="password",
                key=f"emb_api_key_{st.session_state.get('embedding_form_counter', 0)}"
            )
        
        submitted_embedding = st.form_submit_button("Add Embedding Model", width='stretch')
        
        if submitted_embedding:
            # Simple validation
            validation_error = None
            
            if not embedding_model_name or not embedding_model_name.strip():
                validation_error = "Embedding Model name is required!"
            elif not embedding_api_key:
                validation_error = "API Key is required!"
            elif not embedding_endpoint_url:
                validation_error = "Endpoint URL is required!"
            
            if validation_error:
                st.error(f"❌ {validation_error}")
            else:
                # Check if model already exists
                existing_names = [model["name"] for model in available_models]
                if embedding_model_name.strip() in existing_names:
                    st.error(f"❌ Model '{embedding_model_name.strip()}' already exists!")
                else:
                    # Add new embedding model as a separate entry
                    new_embedding_model = {
                        "name": embedding_model_name.strip(),
                        "endpoint": embedding_endpoint_url,
                        "api_key": embedding_api_key,
                        "type": "embedding"
                    }
                    
                    st.session_state.user_preferences["available_models"].append(new_embedding_model)
                    save_user_preferences(st.session_state.user_preferences)
                    
                    # Set success flag and clear form flag
                    st.session_state.embedding_added_success = embedding_model_name.strip()
                    st.session_state.clear_embedding_form = True
                    st.rerun()
    
    # Display success message if an embedding model was just added
    if st.session_state.get("embedding_added_success"):
        st.success(f"✅ Added embedding model '{st.session_state.embedding_added_success}'!")
        del st.session_state.embedding_added_success
    
    st.markdown("---")
    
    st.markdown("### Document Chunking Service")
    
    chunking_url = st.session_state.get("chunking_url", "http://localhost:9876")
    new_chunking_url = st.text_input(
        "Custom Chunking URL",
        value=chunking_url,
        placeholder="http://localhost:9876",
        help="URL for the external document chunking service. Default is the local endpoint.py server."
    )
    
    col_save_url, col_test_url = st.columns([1, 1])
    
    with col_save_url:
        if st.button("Save URL", key="save_chunking_url", width='stretch'):
            st.session_state.chunking_url = new_chunking_url
            st.success("✅ Chunking URL updated successfully!")
    
    with col_test_url:
        if st.button("Test Connection", key="test_chunking_url", width='stretch'):
            try:
                # Test the connection by trying to reach the server
                try:
                    response = requests.get(f"{new_chunking_url.rstrip('/')}", timeout=5)
                    # Even if we get 404, it means the server is responding
                    if response.status_code in [200, 404, 405]:  # 405 = Method Not Allowed is also OK
                        st.success("✅ Connection successful! Server is responding.")
                    else:
                        st.error(f"❌ Connection failed: HTTP {response.status_code}")
                except requests.exceptions.ConnectionError:
                    # If connection fails, try alternative method
                    st.error("❌ Cannot reach chunking service. This is optional - the app will work without it.")
                    st.info("💡 To use the chunking service, start it with: `python endpoint.py`")
                except requests.exceptions.Timeout:
                    st.error("❌ Chunking service test timed out. Service may be slow or unreachable.")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
    
    # Add bottom margin for better spacing
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")   