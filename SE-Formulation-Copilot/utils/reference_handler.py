"""
Reference Handler
Handles source citations and modal dialogs for displaying document chunks.
Used by both main chat and workflow templates.
"""

import streamlit as st

@st.dialog("Citation")
def show_source_modal(filename, chunks, file_index, from_all_sources=False, files_dict=None, message_index=None):
    """Show a modal dialog with source information and all relevant chunks"""
    
    # Add back button if came from "All Sources" dialog
    if from_all_sources and files_dict is not None and message_index is not None:
        if st.button("← Back to All Sources"):
            st.session_state[f"show_all_sources_{message_index}"] = files_dict
            st.rerun()
  
    # Clean filename for display
    display_filename = filename.replace(" (recently uploaded)", "")
    
    # Header
    st.markdown(f"### Source")
    st.markdown(f"**{display_filename}**")
    
    # Light separator with no spacing
    st.markdown('<hr style="margin: 0; border: none; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)
    
    # Show content from chunks
    st.markdown(f"### Content")
    
    # Display each chunk separately with its own relevance score or image thumbnail
    for i, chunk in enumerate(chunks):
        # Check if this is an image source
        if 'image_data' in chunk and chunk.get('image_data'):
            # This is an image - display thumbnail instead of chunk format
            image_data = chunk['image_data']
            
            # Display image info header
            st.markdown(f'<div style="margin-bottom: 8px;"></div>', unsafe_allow_html=True)
            
            # Show image thumbnail in a container
            try:
                import base64
                from PIL import Image
                import io
                
                # Decode base64 image
                img_bytes = base64.b64decode(image_data['base64'])
                image = Image.open(io.BytesIO(img_bytes))
                
                # Display image with max width
                st.image(image, caption=f"{image_data['filename']} ({image_data['width']}×{image_data['height']}, {format_file_size(image_data['size'])})", width='stretch')
                
            except Exception as e:
                # Fallback if image display fails
                st.markdown(f"""
                <div style="
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 0 0 20px 0;
                    text-align: center;
                    color: #666;
                ">📷 Image: {image_data['filename']}<br>
                ({image_data['width']}×{image_data['height']}, {format_file_size(image_data['size'])})<br>
                <small>Error displaying image: {str(e)}</small></div>
                """, unsafe_allow_html=True)
        else:
            # Regular text chunk - show with relevance score
            chunk_score = chunk.get('score', 0)
            chunk_percent = f"{chunk_score * 100:.2f}%"
            
            # Show chunk header with relevance (reduced spacing)
            st.markdown(f'<div style="margin-bottom: 8px;"><strong>Chunk {i+1}</strong> - Relevance: {chunk_percent}</div>', unsafe_allow_html=True)
            
            # Display chunk content as read-only text in a styled container
            chunk_content = chunk.get('content', '')
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 16px;
                margin: 0 0 20px 0;
                max-height: 300px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.4;
                white-space: pre-wrap;
            ">{chunk_content}</div>
            """, unsafe_allow_html=True)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

@st.dialog("All Sources")
def show_all_sources_modal(files_dict, message_index):
    """Show a modal dialog with all source files"""
    # Add CSS for left-aligned buttons
    st.markdown("""
    <style>
    .stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    </style>
    """, unsafe_allow_html=True)

    for i, (filename, chunks) in enumerate(files_dict.items()):
        # Clean filename for display
        display_filename = filename.replace(" (recently uploaded)", "")
        
        # Create clickable citation buttons with proper numbering starting from [1]
        citation_number = i + 1  # Start from [1] for all sources
        
        if st.button(
            f"[{citation_number}] {display_filename}",
            key=f"all_sources_{message_index}_{i}",
            width='stretch'
        ):
            # Store the source info in session state and close this dialog
            st.session_state[f"show_source_{message_index}"] = {
                "filename": filename,
                "chunks": chunks,
                "file_index": citation_number,
                "from_all_sources": True,
                "files_dict": files_dict,
                "message_index": message_index
            }
            st.rerun()

def render_sources_ui(sources, message_index, is_workflow=False):
    """Render compact inline references stacked tightly"""
    if not sources:
        return

    # Different CSS styling for main chat vs workflow templates
    if is_workflow:
        # Workflow template styling - shift right
        st.markdown("""
        <style>
        /* Override global stMarkdownContainer rule for references header - workflow */
        [data-testid="stMarkdownContainer"]:has(.references-header) {
            margin-top: -1rem !important;
            padding-top: 0rem !important;
            margin-left: 1.5rem !important;
        }
        
        /* Collapse vertical spacing between citation buttons - workflow */
        div[data-testid="stElementContainer"][class*="st-key-source_"],
        div[data-testid="stElementContainer"][class*="st-key-show_all_"] {
            margin-top: -0.5rem !important;
            margin-bottom: -0.5rem !important;
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            margin-left: 3.5rem !important;
        }

        /* Keep citation buttons compact - workflow */
        div[data-testid="stElementContainer"][class*="st-key-source_"] .stButton button,
        div[data-testid="stElementContainer"][class*="st-key-show_all_"] .stButton button {
            padding: 2px 6px !important;
            font-size: 0.85rem !important;
            height: auto !important;
            margin: 0 !importantpython;
        }
        </style>
        """, unsafe_allow_html=True)
        header_margin = "2rem"  # Workflow header alignment - shifted right
        header_top_margin = "-10rem"  # Pull closer to chat response for workflows
        header_class = "references-header"  # Class for workflow templates
    else:
        # Main chat styling - original position
        st.markdown("""
        <style>
        /* Override global stMarkdownContainer rule for main chat references header */
        [data-testid="stMarkdownContainer"]:has(.references-header-main) {
            margin-left: -2rem !important;
        }
        
        /* Collapse vertical spacing between citation buttons - main chat */
        div[data-testid="stElementContainer"][class*="st-key-source_"],
        div[data-testid="stElementContainer"][class*="st-key-show_all_"] {great
            margin-top: 0rem !important;
            margin-bottom: -1rem !important;
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            margin-left: 0rem !important;
        }

        /* Keep citation buttons compact - main chat */
        div[data-testid="stElementContainer"][class*="st-key-source_"] .stButton button,
        div[data-testid="stElementContainer"][class*="st-key-show_all_"] .stButton button {
            padding: 2px 6px !important;
            font-size: 0.85rem !important;
            height: auto !important;
            margin: 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        header_margin = "2rem"  # Main chat header alignment - handled by CSS above
        header_top_margin = "0rem"  # Keep normal spacing for main chat
        header_class = "references-header-main"  # Specific class for main chat

    # Group sources
    files_dict = {}
    for chunk in sources:
        filename = chunk.get('filename', 'Unknown')
        files_dict.setdefault(filename, []).append(chunk)

    unique_files = list(files_dict.keys())
    show_count = 2
    additional_count = max(0, len(unique_files) - show_count)

    # Header with dynamic margin based on context
    st.markdown(
        f'<div class="{header_class}" style="margin-top:{header_top_margin} !important;margin-bottom:0rem !important;'
        f'margin-left:{header_margin} !important;padding:0rem !important;'
        'color:#666;font-size:0.9rem;">References from</div>',
        unsafe_allow_html=True
    )   

    # Render first few sources
    for i, filename in enumerate(unique_files[:show_count]):
        # Clean filename for display by removing "(recently uploaded)" text
        display_filename = filename.replace(" (recently uploaded)", "")
        chunks = files_dict[filename]
        
        # Show citation button for all sources (no thumbnails in list)
        if st.button(
            f"[{i+1}] {display_filename}",
            key=f"source_{message_index}_{i+1}",
            width='content'
        ):
            show_source_modal(filename, chunks, i+1)

    # "and X more" button
    if additional_count > 0:
        if st.button(
            f"and {additional_count} more ▼",
            key=f"show_all_{message_index}",
            width='content'
        ):
            show_all_sources_modal(files_dict, message_index)
    
    # Close the margin container    
    # Add consistent bottom spacing after sources
    st.markdown('<div style="margin-bottom: 0.2rem;"></div>', unsafe_allow_html=True)
