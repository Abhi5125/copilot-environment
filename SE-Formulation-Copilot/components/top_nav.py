"""
Top navigation component for Chat JPL application.
Includes settings button and knowledge base selector.
"""

import streamlit as st
from utils.data_persistence import save_user_preferences

import streamlit as st
from utils.data_persistence import save_user_profile

def render_back_to_chat_button():
    if st.button("← Back", key="back_to_chat"):
        st.session_state.current_page = "chat"
        st.query_params.page = "chat"
        st.rerun()

def render_top_nav():
    """Render the top navigation bar with KB selector (if applicable), settings, and profile buttons."""

    current_page = st.session_state.get("current_page", "chat")
    # Only show KB selector for proposal writing and AO compliance checker modes
    show_kb_selector = False
    if current_page not in ["profile", "settings"]:
        if current_page == "proposal_assistant":
            # Only show for writing or ao_comparison sub-modes
            proposal_mode = st.session_state.get("proposal_mode", "main")
            if proposal_mode in ["writing", "ao_comparison"]:
                show_kb_selector = True
        else:
            show_kb_selector = True
    
    # Always use the same column ratios
    col1, col_spacer, col_right = st.columns([6, 3, 1.2])

    # Knowledge Base selector (only render content if allowed)
    with col1:
        if show_kb_selector:
            # knowledge_bases = st.session_state.get("knowledge_bases", [])
            knowledge_bases = []
            kb_options = ["None"] + [kb["name"] for kb in knowledge_bases]

            if kb_options and len(kb_options) > 1:
                # Simple KB selection based on current page
                if current_page == "science_traceability":
                    pref_key = "stm_kb"
                    selector_key = "kb_selector_stm"
                elif current_page == "gate_product_developer":
                    pref_key = "gate_product_kb"
                    selector_key = "kb_selector_gate"
                elif current_page == "heritage_finder":
                    pref_key = "heritage_finder_kb"
                    selector_key = "kb_selector_heritage"
                elif current_page == "proposal_assistant":
                    pref_key = "proposal_assistant_kb"
                    selector_key = "kb_selector_proposal"
                else:
                    # Main chat and other pages
                    pref_key = "main_chat_kb"
                    selector_key = "kb_selector_main"
                
                # Get current selection from preferences
                kb_prefs = st.session_state.user_preferences.get("selected_knowledge_base", {})
                stored_kb = kb_prefs.get(pref_key, kb_options[0] if kb_options else "None")
                
                # Validate stored selection exists in available options
                if stored_kb not in kb_options:
                    stored_kb = kb_options[0] if kb_options else "None"
                    # Update the invalid stored value immediately
                    if "selected_knowledge_base" not in st.session_state.user_preferences:
                        st.session_state.user_preferences["selected_knowledge_base"] = {}
                    st.session_state.user_preferences["selected_knowledge_base"][pref_key] = stored_kb
                    save_user_preferences(st.session_state.user_preferences)
                
                # Callback function for selectbox changes
                def on_kb_change():
                    selected_value = st.session_state[selector_key]
                    # Initialize preferences structure if needed
                    if "selected_knowledge_base" not in st.session_state.user_preferences:
                        st.session_state.user_preferences["selected_knowledge_base"] = {}
                    
                    # Update the specific KB preference
                    st.session_state.user_preferences["selected_knowledge_base"][pref_key] = selected_value
                    save_user_preferences(st.session_state.user_preferences)
                
                # Render selectbox with callback
                selected_kb = st.selectbox(
                    "Knowledge Base:",
                    options=kb_options,
                    index=kb_options.index(stored_kb) if stored_kb in kb_options else 0,
                    key=selector_key,
                    on_change=on_kb_change
                )

    # Right side: Settings + Profile
    with col_right:
        tight1, tight2 = st.columns([0.3, 0.7], gap="small")  # keep them tight & aligned

        with tight1:
            if st.button("⚙️", key="settings_btn"):
                st.session_state.current_page = "settings"
                st.query_params.page = "settings"
                st.rerun()

        with tight2:
            """Render the profile widget with name editing capability"""
            # Get user name from the new profile structure
            user_profile = st.session_state.get("user_profile", {})
            user_name = user_profile.get("name") or "Profile"
            
            # Create clickable profile widget
            profile_clicked = st.button(f'👤 {user_name}', key="profile_btn")

            if profile_clicked:
                st.session_state.current_page = "profile"
                st.query_params.page = "profile"
                st.rerun()


    # CSS to shrink spacing + style settings button
    st.markdown("""
        <style>
        /* Target the specific container causing the margin-top issue */
        div[data-testid="stMarkdownContainer"] div[style*="margin-top: 0rem"] {
            margin-top: 0 !important;
        }
        
        /* More specific targeting for the div with margin-top */
        .stMarkdown > div[style*="margin-top"] {
            margin-top: 0 !important;
        }
        
        /* Target any inline style with margin-top in navigation area */
        [data-testid="stElementContainer"] div[style*="margin-top"] {
            margin-top: 0 !important;
        }
        
        /* Settings button styling - black outline */
        div[class*="st-key-settings_btn"] > .stButton > button,
        div[class*="st-key-profile_btn"] > .stButton > button,
        div[class*="st-key-back_to_chat"] > .stButton > button,
        div[class*="st-key-back_to_main_from_writing"] > .stButton > button,
        div[class*="st-key-back_to_proposal_main_from_writing"] > .stButton > button,
        div[class*="st-key-back_to_review_from_ao"] > .stButton > button,
        div[class*="st-key-back_to_main_from_review"] > .stButton > button,
        div[class*="st-key-back_from_horizontal"] > .stButton > button {
            background: white !important;
            color: black !important;
            border: 1px solid #c5cad2 !important;
            border-radius: 5px !important;
        }

        div[class*="st-key-settings_btn"] > .stButton > button:hover,
        div[class*="st-key-profile_btn"] > .stButton > button:hover,
        div[class*="st-key-back_to_chat"] > .stButton > button:hover,
        div[class*="st-key-back_to_main_from_writing"] > .stButton > button:hover,
        div[class*="st-key-back_to_proposal_main_from_writing"] > .stButton > button:hover,
        div[class*="st-key-back_to_review_from_ao"] > .stButton > button:hover,
        div[class*="st-key-back_to_main_from_review"] > .stButton > button:hover,
        div[class*="st-key-back_from_horizontal"] > .stButton > button:hover {
            background: #f8f9fa !important;
            color: black !important;
            border: 1px solid #c5cad2 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
        }
        </style>
    """, unsafe_allow_html=True)