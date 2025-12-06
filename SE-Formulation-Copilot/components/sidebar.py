"""
Sidebar component for Chat JPL application.
"""

import streamlit as st
from utils.data_persistence import save_conversations, save_user_preferences, add_knowledge_base, generate_kb_id
from utils.vector_db import delete_chat_embeddings
from components.workflow_template import restore_conversation_files

def clear_file_upload_states():
    """Clear all file upload related session state keys"""
    keys_to_remove = []
    for key in st.session_state.keys():
        if (key.startswith("processed_") or 
            key.startswith("chat_file_upload") or 
            key.startswith("proposal_file_upload") or 
            key.startswith("heritage_file_upload") or
            key.startswith("proposal_processed_")):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state[key]

def render_sidebar():
    """Render the complete sidebar with all sections"""
    with st.sidebar:
        render_sidebar_header()
        render_knowledge_bases()
        render_recommended_guides()
        render_conversations()

def render_sidebar_header():
    """Render the SE Copilot header itsin sidebar"""
    # Clean up any cached toggle button state
    if "toggle_sidebar" in st.session_state:
        del st.session_state["toggle_sidebar"]
    
    # Static SE Copilot title with same font as welcome screen but bigger and bold
    st.markdown('<h1 class="main-header" style="font-size: 3rem; font-weight: bold; margin-bottom: 2rem;">SE Copilot</h1>', unsafe_allow_html=True)

    # New Chat button
    if st.button("➕ New Chat", key="new_chat", width='stretch'):
        # Clear all conversation-related state
        st.session_state.current_conversation = None
        st.session_state.current_chat_id = None
        st.session_state.messages = []
        
        # Clear file upload states for all chats
        clear_file_upload_states()
        
        # Clear last conversation preference
        if "user_preferences" in st.session_state:
            st.session_state.user_preferences["last_conversation"] = None
            save_user_preferences(st.session_state.user_preferences)
        
        # Clear specialized chat messages
        if "proposal_writing_messages" in st.session_state:
            st.session_state.proposal_writing_messages = []
        if "ao_comparison_messages" in st.session_state:
            st.session_state.ao_comparison_messages = []
        if "heritage_messages" in st.session_state:
            st.session_state.heritage_messages = []
        
        # Reset specialized chat IDs to create new sessions
        if "proposal_writing_chat_id" in st.session_state:
            del st.session_state.proposal_writing_chat_id
        if "ao_comparison_chat_id" in st.session_state:
            del st.session_state.ao_comparison_chat_id
        if "heritage_chat_id" in st.session_state:
            del st.session_state.heritage_chat_id
        
        # Reset to main chat page
        st.session_state.current_page = "chat"
        st.query_params.page = "chat"
        st.rerun()

def render_knowledge_bases():
    """Render the knowledge bases section with list and add form"""
    st.markdown("---")
    
    # Knowledge Bases header
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🧠 Knowledge Bases</div>', unsafe_allow_html=True)
    
    # List existing knowledge bases
    if st.session_state.knowledge_bases:
        for kb in st.session_state.knowledge_bases:
            st.markdown(f"📚 {kb.get('name', 'Unnamed KB')}")
    else:
        st.markdown('<div class="no-conversations">No knowledge bases yet</div>', unsafe_allow_html=True)
    
    # Add New Knowledge Base form
    with st.expander("➕ Add New Knowledge Base"):
        st.write("DEBUG: KB Form Loaded")  # TEMP: Remove after confirming it shows
        with st.form("new_kb_form", clear_on_submit=True):
            kb_name = st.text_input(
                "Knowledge Base Name",
                placeholder="e.g., Mars 2020 Mission KB",
                key="new_kb_name"
            )
            description = st.text_area(
                "Description",
                placeholder="Brief description of what this knowledge base contains...",
                key="new_kb_description",
                height=100
            )
            
            # New: Project Artifacts selectbox
            project_artifacts_options = [
                "Project-Environment",
                "Fire-Warden",
                "Mission Control",
                "Modeling Environment",
                "Fire cloud"
            ]
            project_artifacts = st.selectbox(
                "Project Artifacts",
                options=project_artifacts_options,
                index=None,
                key="new_kb_project_artifacts",
                help="Select the relevant project artifact"
            )
            
            # New: Artifact Version selectbox
            artifact_version_options = ["v1.1", "v1.2", "v1.3"]
            artifact_version = st.selectbox(
                "Artifact Version",
                options=artifact_version_options,
                index=None,
                key="new_kb_artifact_version",
                help="Select the artifact version"
            )
            
            submitted = st.form_submit_button("✨ Create Knowledge Base", use_container_width=True, type="primary")
            
            if submitted:
                # Validation
                if not kb_name.strip():
                    st.error("Please enter a Knowledge Base Name.")
                elif not description.strip():
                    st.error("Please enter a Description.")
                elif project_artifacts is None:
                    st.error("Please select a Project Artifact.")
                elif artifact_version is None:
                    st.error("Please select an Artifact Version.")
                else:
                    # Create KB data
                    kb_data = {
                        "name": kb_name.strip(),
                        "description": description.strip(),
                        "project_artifacts": project_artifacts,
                        "artifact_version": artifact_version,
                        "id": generate_kb_id()
                    }
                    
                    # Add and save
                    if add_knowledge_base(kb_data):
                        st.success(f"Created Knowledge Base: {kb_name}")
                        # Clear form fields manually for selects (form clear_on_submit handles text)
                        st.session_state.new_kb_project_artifacts = None
                        st.session_state.new_kb_artifact_version = None
                    else:
                        st.error("Failed to create Knowledge Base. Please try again.")
                    
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_recommended_guides():
    """Render the recommended guides section"""
    st.markdown("---")
    
    # Recommended Guides with beautiful styling
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Recommended Guides</div>', unsafe_allow_html=True)
    
    # Simple guide links without duplicating the main content
    guide_links = [
        "Proposal Assistant"
    ]
    
    for guide in guide_links:
        if st.button(guide, width='stretch', key=f"sidebar_{guide}"):
            if guide == "Proposal Assistant":
                # Clear proposal chat state for new chat
                st.session_state.current_conversation = None
                if "proposal_writing_messages" in st.session_state:
                    del st.session_state.proposal_writing_messages
                if "proposal_writing_chat_id" in st.session_state:
                    del st.session_state.proposal_writing_chat_id
                if "ao_comparison_messages" in st.session_state:
                    del st.session_state.ao_comparison_messages
                if "ao_comparison_chat_id" in st.session_state:
                    del st.session_state.ao_comparison_chat_id
                clear_file_upload_states()
                
                # Set proposal mode to main to show the selection page
                st.session_state.proposal_mode = "main"
                st.session_state.current_page = "proposal_assistant"
                st.query_params.page = "proposal_assistant"
            
            else:
                # For any other guides, add chat message and stay on chat
                st.session_state.current_page = "chat"
                st.session_state.messages = []
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Welcome to the {guide}! How can I assist you today?",
                    "sources": []
                })
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_conversations():
    """Render the conversations section"""
    st.markdown("---")
    
    # Conversations with enhanced styling
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Conversations</div>', unsafe_allow_html=True)

    if not st.session_state.conversations:
        st.markdown('<div class="no-conversations">No conversations yet<br>Start chatting to see your history!</div>', unsafe_allow_html=True)
    else:
        # Display conversations in reverse order (newest first)
        for i, conv in enumerate(reversed(st.session_state.conversations)):
            # Calculate the actual index in the original list
            actual_index = len(st.session_state.conversations) - 1 - i
            render_conversation_item(actual_index, conv)

    st.markdown('</div>', unsafe_allow_html=True)

def render_conversation_item(i, conv):
    """Render a single conversation item"""
    # Create columns for conversation button and delete button
    col_conv, col_delete = st.columns([4, 1])
    
    with col_conv:
        # Main conversation button (title only)
        if st.button(
            conv['title'],  # Use title as-is since emojis are already included when saved
            key=f"conv_{i}",
            width='stretch'
        ):
            # Log conversation loading
            print(f"Loaded conversation with ID: {conv.get('chat_id', 'N/A')}")
            
            # Clear file upload states when switching conversations
            clear_file_upload_states()
            
            # Load the selected conversation
            st.session_state.current_conversation = conv['title']
            st.session_state.current_chat_id = conv.get('chat_id')
            
            # Save current conversation to user preferences for restoration on refresh
            if "user_preferences" in st.session_state:
                st.session_state.user_preferences["last_conversation"] = conv['title']
                save_user_preferences(st.session_state.user_preferences)
            
            # Set the appropriate page and messages based on conversation type
            if conv.get('type') == 'proposal_writing':
                print(f"[SIDEBAR] Restoring proposal conversation: {conv.get('title')}")
                st.session_state.current_page = "proposal_assistant"
                st.query_params.page = "proposal_assistant"
                st.session_state.proposal_writing_messages = conv.get('messages', [])
                st.session_state.proposal_writing_chat_id = conv.get('chat_id')
                st.session_state.current_conversation = conv.get('title', None)
                st.session_state["_restoring_proposal_conversation"] = True
                st.session_state.proposal_mode = "writing"
                print(f"[SIDEBAR] Set current_page to: {st.session_state.current_page}")
                print(f"[SIDEBAR] Set proposal_mode to: {st.session_state.proposal_mode}")
                # Restore uploaded files for this conversation
                restore_conversation_files('proposal_writing', conv.get('chat_id'), conv.get('uploaded_files', []))
            elif conv.get('type') == 'ao_comparison':
                st.session_state.current_page = "proposal_assistant"
                st.query_params.page = "proposal_assistant"
                st.session_state.ao_comparison_messages = conv.get('messages', [])
                st.session_state.ao_comparison_chat_id = conv.get('chat_id')
                st.session_state.current_conversation = conv.get('title', None)
                st.session_state["_restoring_ao_conversation"] = True
                st.session_state.proposal_mode = "ao_comparison"
                restore_conversation_files('ao_comparison', conv.get('chat_id'), conv.get('uploaded_files', []))
            else:
                st.session_state.current_page = "chat"
                st.query_params.page = "chat"
                st.session_state.messages = conv.get('messages', [])
                # For regular chat, files are handled differently - you may need to add support if needed
            
            st.rerun()
        
        # Date below the button in smaller text - using CSS to override Streamlit's spacing
        st.markdown(
            f"""
            <style>
            .conversation-date {{
                font-size: 0.75rem !important;
                color: #6b7280 !important;
                margin-top: 0rem !important;
                margin-bottom: 0.5rem !important;
                padding-left: 0.25rem !important;
                line-height: 1 !important;
                position: relative !important;
                top: -0.5rem !important;
            }}
            </style>
            <div class="conversation-date">📅 {conv.get("date", conv.get("timestamp", "Unknown date"))}</div>
            """,
            unsafe_allow_html=True
        )
    
    with col_delete:
        # Delete button
        if st.button("✖", key=f"delete_conv_{i}"):
            if 'chat_id' in conv:
                delete_chat_embeddings(conv['chat_id'])
            if st.session_state.current_conversation == conv['title']:
                st.session_state.current_conversation = None
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                # Clear file upload states when deleting current conversation
                clear_file_upload_states()
                
                # Clear last conversation preference if we're deleting the current one
                if "user_preferences" in st.session_state:
                    st.session_state.user_preferences["last_conversation"] = None
                    save_user_preferences(st.session_state.user_preferences)
                    
            st.session_state.conversations.pop(i)
            save_conversations(st.session_state.conversations)
            st.rerun()