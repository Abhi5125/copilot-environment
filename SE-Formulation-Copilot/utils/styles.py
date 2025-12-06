"""
Streamlit CSS styles for the Chat JPL application.
Centralized styling configuration.
"""

import streamlit as st

def apply_custom_css():
    """Apply all custom CSS styles to the Streamlit app."""
    st.markdown("""
<style>
    /* Import Inter font for consistent typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Remove top spacing and margins */
    .stAppViewContainer, .stDecoration, .stMainBlockContainer {
        margin-top: 0 !important;
        padding-top: 0 !important;
        height: auto !important;
        min-height: 0 !important;
        border-top: none !important;
    }
    
    body, .stApp, .main {
        margin-top: 0 !important;
        padding-top: 0 !important;
        border-top: none !important;
        min-height: 0 !important;
    }
    body > * {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .stApp > * {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .main > * {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    /* If any gap remains, use negative margin to pull content up */
    .main .block-container {
        margin-top: -24px !important;
        padding-right: 3rem !important;
        padding-left: 3rem !important;
    }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
        padding-bottom: 3rem !important;
            padding-right: 3rem !important;
            padding-left: 3rem !important;
            margin-right: 0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        max-width: 100%;
        overflow-x: hidden;
    }

        .stMainBlockContainer {
            padding-right: 3rem !important;
            padding-left: 3rem !important;
            margin-right: 0 !important;
        }
    
    /* Force remove all top spacing */
    .main {
        padding-top: 0 !important;
        margin-top: 0 !important;
        padding-bottom: 3rem !important;
        min-height: auto !important;
    }
    
    .stApp > header {
        background-color: transparent;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    .stApp {
        margin-top: 0 !important;
        padding-top: 0 !important;
        min-height: auto !important;
    }
    
    /* Hide copy to clipboard buttons */
    button[title="Copy to clipboard"] {
        display: none !important;
    }
    
    /* Force sidebar to always be visible */
    .stSidebar,
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: translateX(0) !important;
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fb 100%);
        border-right: 1px solid #e1e8ed;
        transition: all 0.3s ease;
        margin-top: 0px;
        height: 100vh;
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    /* Header Styles */
    .main-header {
        text-align: center;
        background: black;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 10rem;
        font-weight: 300;
        margin-bottom: 0.5rem;
        margin-top: 0;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        text-align: center !important;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
        line-height: 1.6;
        font-weight: 400;
        max-width: 700px;
        margin-left: auto !important;
        margin-right: auto !important;
        padding: 0 1rem;
        display: block !important;
    }
    
    /* Welcome Screen Workflow Images */
    .welcome-content img,
    .guide-card img {
        max-width: 100%;
        width: 100%;
        height: auto !important;
        object-fit: contain;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        margin-bottom: 0.75rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    .welcome-content img:hover,
    .guide-card img:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Style images in columns specifically */
    div[data-testid="column"] img {
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        margin-bottom: 0.75rem !important;
    }
    
    div[data-testid="column"] img:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15) !important;
    }
                
    /* Guide buttons styling - applies to both sidebar and welcome page */
    div[class*="st-key-sidebar_"] > .stButton > button,
    div[class*="st-key-welcome_"] > .stButton > button,
    div[class*="st-key-save_rag"] > .stButton > button,
    div[class*="st-key-delete_kb_"] > .stButton > button,
    div[class*="st-key-save_kb_config_"] > .stButton > button,
    div[class*="st-key-confirm_no_"] > .stButton > button,
    div[class*="st-key-confirm_yes_"] > .stButton > button,
    div[class*="st-key-save_chunking_url"] > .stButton > button,
    div[class*="st-key-test_chunking_url"] > .stButton > button,
    div[class*="st-key-confirm_no_"] > .stButton > button,
    div[class*="st-key-suggest_filters"] > .stButton > button,
    div[class*="st-key-save_llm_config"] > .stButton > button,
    div[class*="st-key-test_full_llm_setup"] > .stButton > button,
    div[class*="st-key-create_universal_kb"] > .stButton > button,
    div[class*="st-key-clear_universal_kb"] > .stButton > button,
    div[class*="st-key-clear_"] > .stButton > button,
    div[class*="st-key-save_api_key"] > .stButton > button,
    div[class*="st-key-test_api_key"] > .stButton > button,
    div[class*="st-key-test_model_"] > .stButton > button,
    div[class*="st-key-send_message_"] > .stButton > button,
    div[class*="st-key-clear_all_conversations_btn"] > .stButton > button,
    div[class*="st-key-cancel_clear_conversations"] > .stButton > button,
    div[class*="st-key-confirm_delete_conversations_btn"] > .stButton > button,
    div[class*="st-key-save_inclusion_settings"] > .stButton > button,
    div[class*="st-key-save_models"] > .stButton > button,
    div[class*="st-key-write_proposal"] > .stButton > button,
    div[class*="st-key-review_proposal"] > .stButton > button,
    div[class*="st-key-check_ao"] > .stButton > button,
    div[class*="st-key-horizontal_review"] > .stButton > button,
    div[class*="st-key-run_horizontal_review"] > .stButton > button,
    div[class*="st-key-use_pasted_proposal"] > .stButton > button,
    div[class*="st-key-download_horizontal_report"] > .stDownloadButton > button,
    div[class*="st-key-provide_different_proposal"] > .stButton > button,
    div[class*="st-key-use_veritas_file"] > .stButton > button,
    div[class*="st-key-use_nanoswarm_file"] > .stButton > button,
    div[class*="st-key-ao_comparison_use_veritas"] > .stButton > button,
    div[class*="st-key-ao_comparison_use_nanoswarm"] > .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        border: none !important;
    }

    /* Hover state for guide buttons, settings page buttons, and profile form submit button */
    div[class*="st-key-sidebar_"] > .stButton > button:hover,
    div[class*="st-key-welcome_"] > .stButton > button:hover,
    div[class*="st-key-save_rag"] > .stButton > button:hover,
    div[class*="st-key-delete_kb_"] > .stButton > button:hover,
    div[class*="st-key-save_kb_config_"] > .stButton > button:hover,
    div[class*="st-key-confirm_no_"] > .stButton > button:hover,
    div[class*="st-key-confirm_yes_"] > .stButton > button:hover,
    div[class*="st-key-save_chunking_url"] > .stButton > button:hover,
    div[class*="st-key-test_chunking_url"] > .stButton > button:hover,
    div[class*="st-key-confirm_no_"] > .stButton > button:hover,
    div[class*="st-key-suggest_filters"] > .stButton > button:hover,
    div[class*="st-key-save_api_key"] > .stButton > button:hover,
    div[class*="st-key-test_api_key"] > .stButton > button:hover,
    div[class*="st-key-create_universal_kb"] > .stButton > button:hover,
    div[class*="st-key-clear_universal_kb"] > .stButton > button:hover,
    div[class*="st-key-clear_"] > .stButton > button:hover,
    div[class*="st-key-test_model_"] > .stButton > button:hover,
    div[class*="st-key-send_message_"] > .stButton > button:hover,
    div[class*="st-key-clear_all_conversations_btn"] > .stButton > button:hover,
    div[class*="st-key-cancel_clear_conversations"] > .stButton > button:hover,
    div[class*="st-key-confirm_delete_conversations_btn"] > .stButton > button:hover,
    div[class*="st-key-save_inclusion_settings"] > .stButton > button:hover,
    div[class*="st-key-save_models"] > .stButton > button:hover,
    div[class*="st-key-write_proposal"] > .stButton > button:hover,
    div[class*="st-key-review_proposal"] > .stButton > button:hover,
    div[class*="st-key-check_ao"] > .stButton > button:hover,
    div[class*="st-key-horizontal_review"] > .stButton > button:hover,
    div[class*="st-key-run_horizontal_review"] > .stButton > button:hover,
    div[class*="st-key-use_pasted_proposal"] > .stButton > button:hover,
    div[class*="st-key-download_horizontal_report"] > .stDownloadButton > button:hover,
    div[class*="st-key-provide_different_proposal"] > .stButton > button:hover,
    div[class*="st-key-use_veritas_file"] > .stButton > button:hover,
    div[class*="st-key-use_nanoswarm_file"] > .stButton > button:hover,
    div[class*="st-key-ao_comparison_use_veritas"] > .stButton > button:hover,
    div[class*="st-key-ao_comparison_use_nanoswarm"] > .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .main .stButton > button:hover:not([key^="delete_conv_"]):not([key*="conv_"]) {
        background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(220, 38, 38, 0.4) !important;
    }

    /* Settings button hover state */
    div[class*="st-key-settings_btn"] > .stButton > button:hover {
        background: #f8f9fa !important;
        color: black !important;
        border: 2px solid #000000 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
    }

    /* Reference/source buttons - reduce padding */
    div[class*="st-key-source_"] > .stButton > button,
    div[class*="st-key-show_all_"] > .stButton > button {
        padding: 0.1rem 0.5rem !important;
        font-size: 0.6rem !important;
        min-height: auto !important;
        height: auto !important;
        line-height: 1.2 !important;
        border: none !important;
    }

    /* Reference/source buttons hover state */
    div[class*="st-key-source_"] > .stButton > button:hover,
    div[class*="st-key-show_all_"] > .stButton > button:hover {
        background: #f8f9fa !important;
        color: black !important;
        border: none !important;
        transform: translateY(-1px) !important;
    }

    /* Delete conversation buttons - override general button styling */
    section[data-testid="stSidebar"] button[key^="delete_conv_"] {
        background: linear-gradient(135deg, #e5e7eb 0%, #d1d5db 100%) !important;
        color: #374151 !important;
        border: 1px solid #d1d5db !important;
        font-size: 0.7rem !important;
        height: 1.2rem !important;
        min-height: 1.2rem !important;
        width: 1.8rem !important;
        min-width: 1.8rem !important;
        max-width: 1.8rem !important;
        padding: 0.1rem !important;
        margin: 0 !important;
        border-radius: 4px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
            
    section[data-testid="stSidebar"] button[key^="delete_conv_"]:hover {
        background: linear-gradient(135deg, #d1d5db 0%, #9ca3af 100%) !important;
        color: #1f2937 !important;
        border-color: #9ca3af !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
    }
    
    
    /* Form Submit Button Styling - Create Knowledge Base */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
        width: 100% !important;
    }

    /* Target the specific button classes from inspect element */
    .stButton .st-emotion-cache-5qfegl,
    button.st-emotion-cache-5qfegl {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Chat Messages */
    .message-user {
        background: #dcf2ff;
        color: #1a1a1a;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem 4rem;
        width: fit-content;
        max-width: 70%;
        margin-left: auto;
        margin-right: 1rem;
        word-wrap: break-word;
        font-weight: 400;
        line-height: 1.4;
    }
    
    .message-assistant {
        background: #f1f1f1;
        color: #1a1a1a;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 1rem 0.5rem 0;
        width: fit-content;
        max-width: 70%;
        word-wrap: break-word;
        line-height: 1.6;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 0.75rem 1.25rem;
        background: #ffffff;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Sidebar styling elements */
    .sidebar-section {
        margin-bottom: 0.01em;
        margin-bottom: 0.001em;
        margin-top: 0.25rem;
    }
    
    /* Reduce spacing around sidebar dividers */
    section[data-testid="stSidebar"] hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Reduce spacing after sidebar titles */
    .sidebar-title {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        color: #1e293b;
        letter-spacing: -0.01rem;
    }
    
    .chat-jpl-title {
        font-size: 3rem;
        font-weight: 500;
        font-size: 3rem;
        background: black;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.75rem;
        text-align: center;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    
    .no-conversations {
        color: #94a3b8;
        font-style: italic;
        font-size: 0.9rem;
        text-align: center;
        padding: 0.75rem 0;
        line-height: 1.4;
    }
    
    /* Delete Button Styling */
    div[data-testid="column"] button[key*="delete_conv_"] {
        background-color: #f9fafb !important;
        color: #6b7280 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        padding: 0.3rem 0.5rem !important;
        width: 100% !important;
        height: 2rem !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="column"] button[key*="delete_conv_"]:hover {
        background-color: #f3f4f6 !important;
        color: #ef4444 !important;
        border-color: #fecaca !important;
    }
    
        background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 8px;
        border: 1px dashed #cbd5e1;
    }
    
    .profile-avatar {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%);
    }
    
    /* Welcome screen animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .welcome-content {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Prevent red hover for sidebar collapse button using data-testid and class */
    div[data-testid="stSidebarCollapseButton"] > button[data-testid="stBaseButton-headerNoPadding"] {
        background: #fff !important;
        color: #1e293b !important;
        border: none !important;
        box-shadow: none !important;
        transition: background 0.2s ease !important;
    }
    div[data-testid="stSidebarCollapseButton"] > button[data-testid="stBaseButton-headerNoPadding"]:hover {
        background: #f3f4f6 !important;
        color: #1e293b !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Text area styling to match text inputs */
    .stTextArea > div > div > textarea {
        padding: 0.75rem 1.25rem !important;
        background: #ffffff !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        font-family: 'Inter', sans-serif !important;

        border: 1px solid #d1d5db !important;  /* light gray outline */
        border-radius: 6px !important;
    }

    
    .stTextArea > div > div > textarea:focus {
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Text input styling consistency */
    .stTextInput > div > div > input {
        padding: 0.75rem 1.25rem !important;
        background: #ffffff !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        border-radius: 6px !important;
        border: 1px solid #d1d5db !important;  /* light gray outline */
        
    }
    
    .stTextInput > div > div > input:focus {
        border-color: transparent !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Hide copy/link icons */
    svg[data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    
    /* Hide any copy-related icons */
    [data-testid="stHeaderActionElements"] svg {
        display: none !important;
    }
    
    /* Hide elements with copy or link functionality */
    button[title*="copy" i], 
    button[title*="link" i],
    svg[class*="copy" i],
    svg[class*="link" i] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

def apply_sidebar_toggle_css(sidebar_expanded):
    """Apply CSS for sidebar toggle functionality."""
    if not sidebar_expanded:
        st.markdown("""
        <style>
        section[data-testid="stSidebar"] {
            transform: translateX(-100%) !important;
            transition: transform 0.3s ease !important;
            width: 280px !important;
            min-width: 280px !important;
            max-width: 280px !important;
        }
        .main .block-container {
            margin-left: 0 !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: none !important;
            transition: all 0.3s ease !important;
        }
        
        /* Recenter main content when sidebar is hidden */
        .main {
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        section[data-testid="stSidebar"] {
            transform: translateX(0) !important;
            transition: transform 0.3s ease !important;
            width: 280px !important;
            min-width: 280px !important;
            max-width: 280px !important;
            overflow-x: visible !important;
            overflow-y: auto !important;
        }
        .main .block-container {
            margin-left: 280px !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        /* Reset main content positioning when sidebar is visible */
        .main {
            display: block !important;
            justify-content: unset !important;
            width: auto !important;
        }
        
        /* Hide floating button when sidebar is visible */
        .floating-sidebar-toggle {
            display: none !important;
        }
        
        /* Modern Source Citation Styles */
        .source-buttons-container {
            margin: 1rem 0;
            padding: 0.5rem 0;
        }
        
        /* Modal dialog styles */
        .stDialog .stTextArea textarea {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
            background-color: #f8f9fa !important;
            border: 1px solid #e9ecef !important;
        }
        
        /* Dialog header styles */
        .stDialog h3 {
            color: #212529 !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Relevance score styling */
        .stDialog .stMarkdown strong {
            color: #fd7e14 !important; /* Orange color for relevance percentage */
        }
        
        /* Remove extra spacing around horizontal rules in proposal assistant */
        [data-testid="stMarkdownContainer"] hr {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Remove spacing from containers that hold horizontal rules */
        [data-testid="stElementContainer"]:has(hr) {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Target specific emotion cache classes that create extra spacing */
        .st-emotion-cache-1vo6xi6 {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .st-emotion-cache-r44huj {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Remove spacing from any stMarkdown container */
        [data-testid="stMarkdownContainer"] {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        }
        
        /* Multiselect styling */
        div[class*="st-key-fy_selector"] .stSelectbox > div > div,
        div[class*="st-key-custom_param_select"] .stSelectbox > div > div {
            border: 2px solid #374151 !important;
            border-radius: 6px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* More aggressive targeting for selectbox components */
        [data-baseweb="select"] {
            border: 2px solid #374151 !important;
            border-radius: 6px !important;
        }
        
        [data-baseweb="select"] > div {
            border: 2px solid #374151 !important;
            border-radius: 6px !important;
        }
                    
        </style>
        """, unsafe_allow_html=True)