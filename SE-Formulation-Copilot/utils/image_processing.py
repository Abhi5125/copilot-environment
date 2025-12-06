"""
Image processing utilities for chat applications
Handles image upload, processing, and integration with vision-capable LLMs
"""

import streamlit as st
import base64
import io
from PIL import Image
import requests
from typing import List, Dict, Any, Optional, Tuple

def is_vision_capable_model(model_name: str) -> bool:
    """Check if a model supports vision/image analysis"""
    vision_models = [
        "claude-3", "claude-3.5-sonnet", "claude-3.7-sonnet",
        "bedrock-claude-3", "bedrock-claude-3.5-sonnet", "bedrock-claude-3.7-sonnet",
        "gemini-pro-vision", "gemini-1.5",
        "gpt-4o", "gpt-4-vision", "gpt-4-turbo", "gpt-5"
    ]
    
    return any(vision_model in model_name.lower() for vision_model in vision_models)

def process_uploaded_image(uploaded_file) -> Optional[Dict[str, Any]]:
    """Process an uploaded image file and prepare it for LLM analysis"""
    try:
        # Read the image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary (removes alpha channel, handles different formats)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (max 2048x2048 for most vision models)
        max_size = 2048
        if image.width > max_size or image.height > max_size:
            # Calculate aspect ratio preserving resize
            ratio = min(max_size / image.width, max_size / image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to base64 for API transmission
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return {
            "filename": uploaded_file.name,
            "size": uploaded_file.size,
            "width": image.width,
            "height": image.height,
            "format": "JPEG",  # We standardize to JPEG
            "base64": img_base64,
            "mime_type": "image/jpeg"
        }
        
    except Exception as e:
        st.error(f"Error processing image {uploaded_file.name}: {str(e)}")
        return None

def create_vision_message_content(text: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create message content that includes both text and images for vision models"""
    content = []
    
    # Add text content
    if text:
        content.append({
            "type": "text",
            "text": text
        })
    
    # Add image content
    for image_data in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_data['mime_type']};base64,{image_data['base64']}"
            }
        })
    
    return content

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def get_image_analysis_prompt(user_message: str, images: List[Dict[str, Any]]) -> str:
    """Generate an appropriate prompt for image analysis"""
    image_count = len(images)
    
    if image_count == 1:
        base_prompt = "I've uploaded an image. "
    else:
        base_prompt = f"I've uploaded {image_count} images. "
    
    if user_message.strip():
        return base_prompt + user_message
    else:
        # Default analysis prompt
        if image_count == 1:
            return base_prompt + "Please analyze this image and describe what you see. Include any relevant technical details, text, diagrams, or notable features."
        else:
            return base_prompt + "Please analyze these images and describe what you see in each one. Include any relevant technical details, text, diagrams, or notable features. If the images are related, please explain how they connect."

def create_image_sources(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create sources entries for uploaded images to display in the Sources panel"""
    sources = []
    
    for i, image_data in enumerate(images):
        sources.append({
            "filename": f"{image_data['filename']}",
            "content": f"Image: {image_data['filename']} ({image_data['width']}×{image_data['height']}, {format_file_size(image_data['size'])})",
            "image_data": image_data  # Include image data for potential display
        })
    
    return sources

def validate_image_file(uploaded_file) -> bool:
    """Validate that uploaded file is a supported image format"""
    supported_formats = ['png', 'jpg', 'jpeg', 'webp']
    
    if uploaded_file.type.startswith('image/'):
        file_ext = uploaded_file.name.split('.')[-1].lower()
        return file_ext in supported_formats
    
    return False