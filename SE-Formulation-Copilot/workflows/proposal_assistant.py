
import streamlit as st
import time
import os
import requests
import re
from datetime import datetime
from utils.model_handler import get_chat_client, get_embeddings_client
from utils.data_persistence import save_conversations
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from components.top_nav import render_back_to_chat_button
from utils.vector_db import create_chat_id
from components.workflow_template import workflow_ui

# Import RAG functions
try:
    from utils.rag import search_vector_db
    from app import retrieve_relevant_chunks, retrieve_relevant_kb_chunks, add_documents_to_chat_embeddings
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    def retrieve_relevant_chunks(*args, **kwargs):
        return []
    def retrieve_relevant_kb_chunks(*args, **kwargs):
        return []
    def add_documents_to_chat_embeddings(*args, **kwargs):
        return False

# System prompts for different modes
PROPOSAL_ASSISTANT_SYSTEM_PROMPT = """
You are a Proposal Assistant designed to guide users through two possible workflows:

- Option 1: Write a new proposal (Proposal Development Assistant flow)
- Option 2: Review an existing proposal (Proposal-AO Compliance Checker)

The user will already have seen a welcome screen outside this prompt and will choose one of the two options as their first input.
You must detect whether they chose Option 1 or Option 2 and then follow the appropriate workflow below.
Do NOT mix workflows. Always stay consistent with the option chosen until the user finishes or explicitly changes paths.

====================================================
OPTION 1: WRITE A NEW PROPOSAL (Proposal Development Assistant)
====================================================

Step 1 - Mission Concept Summary
- Expect the first message to be the mission summary.
- Acknowledge with:
"Thank you for providing the mission summary for [Mission Name]. Let's move to the next step."

Step 2 - Announcement of Opportunity (AO)
- Ask:
"Please attach or specify the Announcement of Opportunity (AO) relevant to your proposal."
- Wait for upload.
- Acknowledge:
"Great! I will now proceed to the next step."

Step 3 - Identify Requirements to Extract
- Ask:
"Which requirements do you want to extract from the AO? Please indicate a section name or number."
- Extract and present the requested section(s) EXACTLY as written in the AO in a polished format with headers. Do not paraphrase.
- Then ask EXACTLY:
"Would you like me to generate an example response for a specific requirement or check your draft section against the requirement(s)?"

Step 4 - Generate Example Responses
- If user chose example response generation:
  - Use mission details from Step 1 to create a tailored, compliant example response. Fill in missing details with creativity.
  - Do NOT use any bullet points or numbered lists. Everything should be in paragraph format.
  - IMPORTANT <<<YOU MUST INCLUDE THIS AT THE TOP>>>: Add the note
    "The content below serves as an example template. Please modify content to align with your mission."
  - IMPORTANT <<<YOU MUST INCLUDE THIS AT THE TOP>>>: Ask user if they want to proceed with uploading a draft section and checking its compliance with the extracted requirements.

Step 5 - Draft Review & Compliance Check
- If user uploads draft section, ask:
"Please upload or paste your draft section. I will check it for completeness against the extracted requirements."
- Provide:
  * Compare draft against ALL EXTRACTED requirements 1 by 1, with the direct quote of the requirement above each analysis for reference.
  * Compliance score (0–100%)
  * Gap analysis (list missing elements vs AO requirement)
  * Suggested improvements (clear, actionable edits)
- Be very thorough, detailed, and STRICT.
- Make the response very polished and easy to read.
- <<VERY IMPORTANT>> DO NOT include any <br> tags in the response at all. 

Step 6 - Iterate & Finalize
- Guide the user through revisions until they are satisfied.
- After each step, always ask if they would like to move on to the next.

Step 7 - Move to Next Section
- Ask:
"Would you like to proceed to the next section of the AO or finalize the proposal package?"

====================================================
OPTION 2: REVIEW AN EXISTING PROPOSAL (Proposal-AO Compliance Checker)
====================================================

Step 1 - Announcement of Opportunity (AO)
- Expect the first message to be the AO.
- Acknowledge with:
"Thank you for providing your Announcement of Opportunity. Let's move to the next step."

Step 2 - Identify Requirements to Extract
- Ask:
"Which requirements do you want to extract from the AO? Please indicate a section name or number."
- Extract and present the requested section(s) EXACTLY as written in the AO. Do not paraphrase.

Step 3 - Upload Proposal for Comparison
- Ask:
"Please upload your proposal document for comparison against the AO."
- Wait for upload.
- Provide compliance analysis for each requirement:
  * Compare draft against ALL EXTRACTED requirements 1 by 1, with the direct quote of the requirement above each analysis for reference.
  * Compliance score (0–100%)
  * Gap analysis (list missing elements vs AO requirement)
- Be very thorough, detailed, and STRICT.
- Make the response very polished and easy to read.
- <<VERY IMPORTANT>> DO NOT include any <br> tags in the response at all. 
- Acknowledge with:
"Great! I will now proceed to the next step."

Step 4 - Iterate & Finalize
- Guide the user through revisions until the section fully meets AO requirements.

Step 5 - Move to Next Section
- Ask:
"Would you like to proceed to the next section of the AO or finalize the proposal package?"

====================================================
REFERENCE MATERIAL
====================================================

Index of AO Sections:
1. Description of Opportunity
   1.1 Introduction
   1.2 NASA's Policies on Harassment and Discrimination
   1.3 NASA Safety Priorities
2. AO Objectives
   2.1 NASA Strategic Goals
   2.2 Discovery Program Goals and Objectives
   2.3 Discovery Program Background
3. Proposal Opportunity Period and Schedule
4. Policies Applicable to this AO
   4.1 NASA Management Policies
   4.2 Participation Policies
   4.3 Cost Policies
   4.4 Data and Sample Return Policies and Requirements
   4.5 Intellectual Property Rights
   4.6 Project Management Policies
5. Requirements and Constraints
   5.1 General Requirements
   5.2 Mission Requirements
   5.3 Design Requirements
   5.4 Schedule Requirements
6. Evaluation Criteria and Process
7. AO Point of Contact and Pre-Proposal Conference Information
8. Proposal Preparation Instructions
   8.1 General Instructions
   8.2 Step-1 Proposal Format
   8.3 Step-2 Proposal Format
9. Required Appendices
10. References

"""


PROPOSAL_WELCOME_MESSAGE = """
🚀 Welcome to the Proposal Assistant!  

I'm here to help you **create, review, and improve mission proposals**.  
Whether you're working on **science objectives, technical requirements, budget planning, or proposal structure**, I can provide expert guidance based on **JPL best practices** and **NASA standards**.  

**Please choose one of the following options to get started:**

**Option 1:** Write a new proposal  
**Option 2:** Review an existing proposal
"""

AO_WELCOME_MESSAGE = """Welcome to the AO Compliance Checker! I'll help you compare your proposal against Announcement of Opportunity requirements to ensure full compliance.

Please upload the AO document you want to compare your proposal against."""

AO_COMPARISON_SYSTEM_PROMPT = """
You are a Proposal Review Assistant specialized in comparing proposals against Announcement of Opportunity (AO) requirements.

Step 1 - Mission Concept Summary
User will start off by an AO, expect this as first message.
Acknowledge with:
"Thank you for providing your Announcement of Opportunity. Let's move to the next step."

Step 2 - Identify Requirements to Extract
Ask:

"Which requirements do you want to extract from the AO? Please indicate a section name or number."
Extract and present the requested section(s) EXACTLY as written in the AO, do not paraphrase.

Step 3 - Upload Proposal for Comparison
Ask:

"Please upload your proposal document for comparison against the AO."
Wait for upload.

Provide a compliance analysis for to each requirement, and quote the actual requirement above each analysis for reference.

Acknowledge:
"Great! I will now proceed to the next step."

Step 4 - Generate Example Responses
Ask:

"Would you like me to generate an example response for a specific requirement or check your draft section against the requirement(s)?"
If example requested:

Use mission details from Step 3 to create a tailored, compliant example response. Give very thorough and detailed examples.

Step 5 - Iterate & Finalize
Guide the user through revisions until the section fully meets AO requirements.

Step 6 - Move to Next Section
Ask:

"Would you like to proceed to the next section of the AO or finalize the proposal package?"


Index:
1. Description of Opportunity
1.1 Introduction
1.2 NASA's Policies on Harassment and Discrimination
1.3 NASA Safety Priorities
2. AO Objectives
2.1 NASA Strategic Goals
2.2 Discovery Program Goals and Objectives
2.3 Discovery Program Background
3. Proposal Opportunity Period and Schedule
4. Policies Applicable to this AO
4.1 NASA Management Policies
4.2 Participation Policies
4.3 Cost Policies
4.4 Data and Sample Return Policies and Requirements
4.5 Intellectual Property Rights
4.6 Project Management Policies
5. Requirements and Constraints
5.1 General Requirements
5.2 Mission Requirements
5.3 Design Requirements
5.4 Schedule Requirements
6. Evaluation Criteria and Process
7. AO Point of Contact and Pre-Proposal Conference Information
8. Proposal Preparation Instructions
8.1 General Instructions
8.2 Step-1 Proposal Format
8.3 Step-2 Proposal Format
9. Required Appendices
10. References

Keywords: proposal, announcement of opportunity, AO, Discovery program, mission concept, requirements, compliance, gap analysis, draft review, NASA, JPL
"""

HORIZONTAL_REVIEW_SYSTEM_PROMPT = """
You are a Proposal Consistency Checker performing precise horizontal review analysis.

Your primary task is to verify that values for specific terms are completely consistent throughout the entire proposal document.

For each term you analyze:

1. All instances: Locate every mention of the term and its value and display them in a table. Include a column of the exact quote the value was found in Mention the section name header for each instance of the term and its value.
The table column headers should be: Section Name, Quoted Text, Extracted Value

2. Compare for consistency: Compare each value of that term with each other to check if they match exactly, flagging contracitions. Don't include metrics like contingency % just be straight forward.

3. Final Assessment: Clearly state whether values are:
   - CONSISTENT: All values match across the document
   - INCONSISTENT: Contradictory values found (specify exactly what contradicts)

Be extremely precise and thorough. Focus on numerical values, measurements, and specifications. If values are consistent, confirm this clearly. If inconsistent, highlight exactly what contradicts and where.
Always include "the_term_name Consistency Analysis" as the top of your response as a title
"""

def proposal_assistant_ui():
    """Main UI function for the new Proposal Assistant workflow"""

    # Initialize proposal mode if not set
    if "proposal_mode" not in st.session_state:
        st.session_state.proposal_mode = "main"

    # Handle different modes
    if st.session_state.proposal_mode == "main":
        # Add back to chat button only on main screen
        render_back_to_chat_button()
        st.markdown(
        """
        <div style="margin-top:0; padding-top:0">
            <hr style="margin-top:0; margin-bottom:0.25rem; border:1px solid #e5e7eb;" />
        </div>
        """,unsafe_allow_html=True)
        render_main_selection()
    elif st.session_state.proposal_mode == "writing":
        render_proposal_writing()
    elif st.session_state.proposal_mode == "horizontal_review":
        render_horizontal_review()

def render_main_selection():
    """Render the main selection screen with two primary options"""
    # Only print once per actual visit to main selection page
    if not st.session_state.get('_proposal_main_ui_logged', False):
        print(f"[Proposal Assistant] Rendered UI")
        st.session_state._proposal_main_ui_logged = True
    
    # Create two columns for the main options
    col1, col2 = st.columns(2)
    
    with col1:
        # Add slight spacing to move button down

        st.markdown("### ✍️ Get Assistance Writing a Proposal")
        st.markdown("""
        Step-by-step guide to creating a NASA AO-compliant mission proposal.
        
        **What I'll help with:**
        - AO requirement extraction
        - Example response generation
        - Draft section review
        - Compliance checking
        """)

        st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
        if st.button("Get Assistance Writing a Proposal", width="stretch", key="write_proposal"):
            st.session_state.proposal_mode = "writing"
            # Clear the reset flag to allow mode changes
            if "reset_proposal_mode" in st.session_state:
                del st.session_state.reset_proposal_mode
            st.rerun()
    
    with col2:
        st.markdown("### 🔄 Perform Horizontal Review")
        st.markdown("""
        Check for consistency of specific terms and concepts throughout your proposal document.
        
        **What I'll help with:**
        - Upload your proposal
        - Enter terms to check (e.g., Mass, Power, Data volume, Pointing control, Agility)
        - Get detailed consistency analysis
        - Identify and resolve contradictions
        """)
        
        # Add spacing to align with left column (which has 4 bullet points vs 4)
        st.markdown('<div style="height: 1.95rem;"></div>', unsafe_allow_html=True)

        if st.button("Perform Horizontal Review", width="stretch", key="horizontal_review"):
            st.session_state.proposal_mode = "horizontal_review"
            # Set flag to ensure fresh start when entering horizontal review
            st.session_state.horizontal_review_fresh_start = True
            # Clear the reset flag to allow mode changes
            if "reset_proposal_mode" in st.session_state:
                del st.session_state.reset_proposal_mode
            st.rerun()
    
    # Add bottom spacing to prevent page cutoff
    st.markdown('<div style="height: 5rem;"></div>', unsafe_allow_html=True)

def render_proposal_writing():
    """Render the proposal writing workflow using workflow template"""
    print(f"[Proposal Writer] Rendered UI")
    # Use the workflow template for the proposal writing interface
    workflow_ui(
        workflow_name="proposal_writing",
        title="Proposal Writing Assistant",
        description="Step-by-step guidance for creating mission proposals",
        system_prompt=PROPOSAL_ASSISTANT_SYSTEM_PROMPT,
        welcome_message=PROPOSAL_WELCOME_MESSAGE,
        messages_key="proposal_writing_messages",
        chat_id_key="proposal_writing_chat_id",
        input_key="proposal_writing_input_key",
        generating_key="generating_proposal_writing_response"
    )

def render_horizontal_review():
    """Render the horizontal review interface"""
    print(f"[Horizontal Review] Rendered UI")
    # Custom back button for proposal assistant
    if st.button("← Back", key="back_from_horizontal"):
        st.session_state.proposal_mode = "main"
        # Set flag to ensure fresh start when returning to horizontal review
        st.session_state.horizontal_review_fresh_start = True
        st.rerun()
    
    st.markdown(
    """
    <div style="margin-top:0; padding-top:0">
        <hr style="margin-top:0; margin-bottom:0.25rem; border:1px solid #e5e7eb;" />
    </div>
    """,unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center;'>Horizontal Consistency Review</h1>", unsafe_allow_html=True)
    
    # Add custom CSS to limit tab underline width
    st.markdown("""
    <style>
    /* Remove static underline from selected tab, keep only the sliding indicator */
    .stTabs [data-baseweb="tab-border"] {
        width: fit-content !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        border-bottom: 2px solid transparent !important;
        transition: border-bottom 0.2s ease;
        position: relative;
        z-index: 1;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: flex-start;
    }
    /* Style the sliding underline indicator only */
    .stTabs [data-baseweb="tab-highlight"] {
        height: 2px !important;
        background: #ff6b6b !important;
        border-radius: 1px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if we're freshly entering horizontal review mode (reset session state for fresh start)
    if "horizontal_review_fresh_start" not in st.session_state:
        st.session_state.horizontal_review_fresh_start = True
    
    if st.session_state.horizontal_review_fresh_start:
        # Clear all horizontal review session state variables for a fresh start
        st.session_state.horizontal_proposal_uploaded = False
        st.session_state.horizontal_proposal_content = ""
        st.session_state.horizontal_review_results = {}
        # Create a unique chat ID for this horizontal review session
        if "horizontal_review_chat_id" not in st.session_state:
            st.session_state.horizontal_review_chat_id = f"horizontal_review_session_{int(time.time())}"
        st.session_state.horizontal_review_fresh_start = False
    
    # Initialize session state for horizontal review (if not already set)
    if "horizontal_proposal_uploaded" not in st.session_state:
        st.session_state.horizontal_proposal_uploaded = False
    if "horizontal_proposal_content" not in st.session_state:
        st.session_state.horizontal_proposal_content = ""
    if "horizontal_review_results" not in st.session_state:
        st.session_state.horizontal_review_results = {}
    if "horizontal_uploaded_filename" not in st.session_state:
        st.session_state.horizontal_uploaded_filename = None
    
    # Step 1: Upload proposal
    if not st.session_state.horizontal_proposal_uploaded:        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["Upload File", "Paste Text"])
        
        with tab1:
            st.markdown("Upload your proposal document to check for consistency across terms and values.")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'md'],
            )
            
            if uploaded_file is not None:
                # Check if this is a new file (different from previously uploaded)
                if (st.session_state.horizontal_uploaded_filename != uploaded_file.name and 
                    st.session_state.horizontal_uploaded_filename is not None):
                    # Clear previous content when uploading a new file
                    st.session_state.horizontal_proposal_content = ""
                    st.session_state.horizontal_review_results = {}
                    # Clear previous embeddings reference for this specific horizontal review session
                    st.session_state.horizontal_review_chat_id = f"horizontal_review_session_{int(time.time())}"
                    st.info(f"🔄 Cleared previous content. Now processing new file: {uploaded_file.name}")
                
                # Process the uploaded file
                if process_document_content(uploaded_file):
                    st.session_state.horizontal_uploaded_filename = uploaded_file.name
                    st.session_state.horizontal_proposal_uploaded = True
                    st.rerun()
        
        with tab2:
            proposal_text = st.text_area(
                "Paste your proposal text directly into the text area below:",
                placeholder="Paste your proposal content here...",
                height=300,
            )
            
            if st.button("Use Pasted Text", key="use_pasted_proposal"):
                if proposal_text.strip():
                    # Process pasted text for RAG
                    try:
                        # Create chunks from pasted text (simple paragraph splitting)
                        paragraphs = [p.strip() for p in proposal_text.split('\n\n') if p.strip()]
                        
                        # Format chunks for vector storage
                        formatted_chunks = []
                        chunk_texts = []  # For embedding function
                        for i, paragraph in enumerate(paragraphs):
                            chunk_data = {
                                'text': paragraph,
                                'chunk_id': i,
                                'source': 'pasted_text.txt'
                            }
                            formatted_chunks.append(chunk_data)
                            chunk_texts.append(paragraph)  # Text only for embeddings
                        
                        print(f"[HORIZONTAL_REVIEW] Processing {len(formatted_chunks)} chunks from pasted text")
                        
                        # Add to chat embeddings for RAG retrieval
                        # Ensure we have a current chat ID for this horizontal review session
                        if "horizontal_review_chat_id" not in st.session_state:
                            st.session_state.horizontal_review_chat_id = f"horizontal_review_pasted_{int(time.time())}"
                        
                        # Add chunks to chat embeddings
                        with st.spinner(f"Embedding {len(formatted_chunks)} chunks for RAG search..."):
                            success = add_documents_to_chat_embeddings(
                                st.session_state.horizontal_review_chat_id, 
                                chunk_texts,  # Pass text list, not dict list
                                'pasted_text.txt'
                            )
                            
                            if success:
                                pass
                            else:
                                st.warning("⚠️ Vector embedding failed, using direct text analysis")
                    
                    except Exception as e:
                        st.warning(f"Embedding failed: {str(e)}, using direct text analysis")
                    
                    # Store full text as fallback
                    st.session_state.horizontal_proposal_content = proposal_text.strip()
                    st.session_state.horizontal_proposal_uploaded = True
                    st.success("✅ Proposal text loaded successfully!")
                    st.rerun()
                else:
                    st.error("Please paste some proposal text before proceeding.")
    
    else:
        # Step 2: Enter terms to check
        st.markdown("### Step 2: Enter Terms to Check")
        st.markdown("Enter comma-separated terms you want to check for consistency (e.g., Mass, Power, Data Volume)")
        
        terms_input = st.text_input("Enter terms to check for consistency", label_visibility="collapsed", placeholder="e.g., Mass, Power, Data Volume, Pointing Control, Agility")

        if st.button("Run Consistency Check", key="run_horizontal_review"):
            if terms_input.strip():
                terms = [term.strip() for term in terms_input.split(',') if term.strip()]
                run_horizontal_review(terms)
            else:
                st.error("Please enter at least one term to check.")
        
        # Display results if available
        if st.session_state.horizontal_review_results:
            display_horizontal_results()
    
    # Add bottom spacing
    st.markdown('<div style="height: 5rem;"></div>', unsafe_allow_html=True)

def process_document_content(uploaded_file):
    """Process document content using the chunking endpoint and embed into vector database"""
    try:
        # Send file to chunking endpoint for processing
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        chunking_url = st.session_state.get("chunking_url", "http://localhost:9876")
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            response = requests.post(f"{chunking_url}/process", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list):
                    # Check if it's a list of objects with page_content
                    if result and isinstance(result[0], dict) and "page_content" in result[0]:
                        chunks = [item["page_content"] for item in result]
                    else:
                        chunks = result
                elif isinstance(result, dict):
                    chunks = result.get("chunks", [])
                else:
                    chunks = []
                
                # Format chunks for vector storage
                formatted_chunks = []
                chunk_texts = []  # For embedding function
                for i, chunk_text in enumerate(chunks):
                    chunk_data = {
                        'text': chunk_text,
                        'chunk_id': i,
                        'source': uploaded_file.name
                    }
                    formatted_chunks.append(chunk_data)
                    chunk_texts.append(chunk_text)  # Text only for embeddings
                
                print(f"[HORIZONTAL_REVIEW] Processing {len(formatted_chunks)} chunks for {uploaded_file.name}")
                
                # Add to chat embeddings for RAG retrieval
                # Ensure we have a current chat ID for this horizontal review session
                if "horizontal_review_chat_id" not in st.session_state:
                    st.session_state.horizontal_review_chat_id = f"horizontal_review_{int(time.time())}"
                
                # Add chunks to chat embeddings
                with st.spinner(f"Embedding {len(formatted_chunks)} chunks for RAG search..."):
                    success = add_documents_to_chat_embeddings(
                        st.session_state.horizontal_review_chat_id, 
                        chunk_texts,  # Pass text list, not dict list
                        uploaded_file.name
                    )
                    
                    if success:
                        pass
                    else:
                        st.warning("⚠️ Vector embedding failed, using direct text analysis")
                
                # Combine all chunks into a single document (fallback)
                document_content = "\n\n".join([str(chunk) for chunk in chunks])
                st.session_state.horizontal_proposal_content = document_content                
                return True
            else:
                st.error(f"Error processing {uploaded_file.name}: {response.text}")
                return False
                
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False

def run_horizontal_review(terms):
    """Run the horizontal review for each term using RAG"""
    
    # Get chat client using proposal_assistant workflow configuration
    try:
        kb_prefs = st.session_state.user_preferences.get("selected_knowledge_base", {})
        selected_kb_name = kb_prefs.get("proposal_assistant_kb", "None")
        selected_kb = None if selected_kb_name == "None" else selected_kb_name
        chat_client = get_chat_client("proposal_assistant", selected_kb=selected_kb)
    except Exception as e:
        st.error(f"Error connecting to chat client: {str(e)}")
        return
    
    # Get RAG settings from user preferences
    rag_settings = st.session_state.get("rag_settings", {})
    # Allow more chunks for horizontal review while still maintaining context limits
    max_top_k = 25  # Increased limit for better consistency checking
    user_top_k = rag_settings.get("top_k_reranker" if rag_settings.get("reranking_engine", False) else "top_k", 20)
    top_k = min(max_top_k, user_top_k)  # Use smaller value
    
    print(f"[HORIZONTAL_REVIEW] Using RAG settings - top_k: {top_k} (limited from {user_top_k}), reranking: {rag_settings.get('reranking_engine', False)}")
    
    # Create progress bar container
    progress_container = st.empty()
    
    with progress_container.container():
        progress_bar = st.progress(0)
    
    results = {}
    
    for i, term in enumerate(terms):
        # Use spinner for each individual term check
        with st.spinner(f"Checking consistency for: {term}"):
            progress_bar.progress((i + 1) / len(terms))
            
            try:
                # Use RAG to retrieve relevant chunks for this term
                relevant_chunks = []
                
                # Check if we have a horizontal proposal KB (from pre-chunked files)
                if hasattr(st.session_state, 'horizontal_proposal_kb'):
                    print(f"[HORIZONTAL_REVIEW] Searching in KB: {st.session_state.horizontal_proposal_kb}")
                    # Regular KB search
                    relevant_chunks = retrieve_relevant_kb_chunks(term, st.session_state.horizontal_proposal_kb, k=top_k)
                
                # If no KB or not enough chunks, fall back to uploaded file embeddings
                if not relevant_chunks and st.session_state.get('horizontal_review_chat_id'):
                    print(f"[HORIZONTAL_REVIEW] Searching in uploaded files for chat_id: {st.session_state.horizontal_review_chat_id}")
                    relevant_chunks = retrieve_relevant_chunks(term, st.session_state.horizontal_review_chat_id, k=top_k)
                
                print(f"[HORIZONTAL_REVIEW] Retrieved {len(relevant_chunks)} chunks for term: {term}")
                
                # Build context from relevant chunks with RAG preferences
                if relevant_chunks:
                    context_chunks = []
                    total_chars = 0
                    
                    # Use user's RAG preferences instead of hardcoded limits
                    full_context_mode = rag_settings.get("full_context_mode", False)
                    max_chunks_to_use = len(relevant_chunks)  # Use all retrieved chunks by default
                    
                    if full_context_mode:
                        print(f"[HORIZONTAL_REVIEW] Full context mode enabled - using all {len(relevant_chunks)} chunks")
                        # In full context mode, use all chunks without character limits
                        max_context_chars = float('inf')
                    else:
                        # Calculate reasonable context limit based on model and user preferences
                        # Allow for much larger context since modern models can handle it
                        max_context_chars = 50000  # Increased from 15000 to respect user's 25-chunk preference
                        print(f"[HORIZONTAL_REVIEW] Standard mode - using up to {max_chunks_to_use} chunks with {max_context_chars} char limit")
                    
                    for i, chunk in enumerate(relevant_chunks):
                        chunk_content = chunk.get('content', chunk.get('chunk', ''))
                        filename = chunk.get('filename', 'Unknown')
                        score = chunk.get('score', 0.0)
                        
                        # Only truncate chunks in standard mode if they're extremely long
                        if not full_context_mode and len(chunk_content) > 3000:
                            chunk_content = chunk_content[:3000] + "... [truncated for context management]"
                        
                        chunk_text = f"[Score: {score:.3f}] Document: {filename}\nContent: {chunk_content}"
                        
                        # Check context limits only if not in full context mode
                        if not full_context_mode and total_chars + len(chunk_text) > max_context_chars:
                            print(f"[HORIZONTAL_REVIEW] Reached context limit at {len(context_chunks)} chunks ({total_chars} chars)")
                            break
                        
                        context_chunks.append(chunk_text)
                        total_chars += len(chunk_text)
                        
                        # Early exit if we've processed all requested chunks
                        if i + 1 >= max_chunks_to_use:
                            break
                    
                    context = "\n\n---\n\n".join(context_chunks)
                    print(f"[HORIZONTAL_REVIEW] Built context with {len(context_chunks)}/{len(relevant_chunks)} chunks, {total_chars} characters")
                    
                    # Create RAG-enhanced prompt with size management
                    prompt = f"""
Check if the value for "{term}" is completely consistent within the provided document excerpts.

Instructions:
1. Find ALL mentions of "{term}" and related concepts throughout the excerpts
2. Extract the specific values, numbers, specifications, or descriptions associated with "{term}"
3. Compare all instances to check for consistency
4. Identify any contradictions where different values are given for the same parameter
5. Provide specific examples with exact quotes and reference the document sources
6. If values are consistent, confirm this clearly
7. If inconsistent, highlight the conflicting values and where they appear

Example: If "{term}" is "mass" and one excerpt says "10 kg" but another says "15 kg" for the same component, this is an inconsistency that must be flagged.

Relevant Document Excerpts ({len(context_chunks)} excerpts analyzed):
{context}

Focus specifically on checking consistency of values for "{term}". Provide a clear assessment of whether the values are consistent or contradictory across all provided excerpts.
"""
                else:
                    # Fallback to limited document analysis if no relevant chunks found
                    print(f"[HORIZONTAL_REVIEW] No relevant chunks found for {term}, using limited document analysis")
                    
                    # Truncate document content to avoid context window issues
                    document_content = st.session_state.horizontal_proposal_content
                    if len(document_content) > 10000:
                        document_content = document_content[:10000] + "\n... [Document truncated to avoid context window limits]"
                    
                    prompt = f"""
Check if the value for "{term}" is completely consistent within the document excerpt.

Instructions:
1. Find ALL mentions of "{term}" and related concepts throughout the document
2. Extract the specific values, numbers, specifications, or descriptions associated with "{term}"
3. Compare all instances to check for consistency
4. Identify any contradictions where different values are given for the same parameter
5. Provide specific examples with exact quotes and their locations
6. If values are consistent, confirm this clearly
7. If inconsistent, highlight the conflicting values and where they appear

Example: If "{term}" is "mass" and the document says "10 kg" in one place but "15 kg" in another place for the same component, this is an inconsistency that must be flagged.

Document excerpt to analyze (limited to avoid context window issues):
{document_content}

Focus specifically on checking consistency of values for "{term}". Provide a clear assessment of whether the values are consistent or contradictory.
"""
                
                # Make API call with file context if available
                enhanced_system_prompt = HORIZONTAL_REVIEW_SYSTEM_PROMPT
                if st.session_state.get("last_uploaded_file"):
                    file_context_info = f"\n\nIMPORTANT FILE CONTEXT: The most recent file uploaded was '{st.session_state.last_uploaded_file}'. If the user asks about the contents of a file arbitrarily, they are referring to this one."
                    enhanced_system_prompt = HORIZONTAL_REVIEW_SYSTEM_PROMPT + file_context_info
                
                messages = [
                    SystemMessage(content=enhanced_system_prompt),
                    HumanMessage(content=prompt)
                ]
                
                response = chat_client.invoke(messages)
                results[term] = response.content
                
            except Exception as e:
                results[term] = f"Error analyzing {term}: {str(e)}"
            
            time.sleep(0.5)  # Small delay to prevent rate limiting
    
    # Clear the progress bar
    progress_container.empty()
    
    # Store results
    st.session_state.horizontal_review_results = results

def display_horizontal_results():
    """Display the results of the horizontal review"""

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Consistency Check Results")
    
    for term, result in st.session_state.horizontal_review_results.items():
        with st.expander(f"Results for: {term}", expanded=True):
            if isinstance(result, str) and result.startswith("Error"):
                st.error(result)
            else:
                st.markdown(result)
    
    # Create columns for buttons - make them smaller and closer together
    col1, col2, col3 = st.columns([0.3, 0.3, 0.4])
    
    with col1:
        # Generate clean text report for download
        report_content = ""
        for term, result in st.session_state.horizontal_review_results.items():
            report_content += f"Results for: {term}\n\n"
            report_content += f"{result}\n\n"
            report_content += "---\n\n"
        
        # Direct download button
        st.download_button(
            label="Download Results Report",
            key="download_horizontal_report",
            data=report_content,
            file_name=f"horizontal_review_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            width='stretch'
        )
    
    with col2:
        # Option to provide a new proposal
        if st.button("Provide Different Proposal", key="provide_different_proposal", width='stretch'):
            st.session_state.horizontal_proposal_uploaded = False
            st.session_state.horizontal_proposal_content = ""
            st.session_state.horizontal_review_results = {}
            st.session_state.horizontal_uploaded_filename = None
            # Create a new chat ID for the new proposal
            st.session_state.horizontal_review_chat_id = f"horizontal_review_{int(time.time())}"
            st.rerun()
    
    # col3 is empty spacer

    # Add bottom spacing
    st.markdown('<div style="height: 5rem;"></div>', unsafe_allow_html=True)
