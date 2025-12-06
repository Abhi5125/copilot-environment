"""
Workflow Model Management Utilities
Handles model selection for different workflow tools with knowledge base overrides.
"""

import streamlit as st
from utils.model_handler import get_chat_client, CustomEndpointClient
from utils.data_persistence import load_user_preferences


def get_workflow_model_client(workflow_name, temperature=0.7, selected_kb=None):
    """
    Get the appropriate model client for a workflow tool.
    
    Args:
        workflow_name: One of 'main_chat', 'heritage_finder', 
                      'science_traceability_matrix', 'gate_product_developer'
        temperature: Temperature setting for the model
        selected_kb: Selected knowledge base name (for workflows that support KB override)
    
    Returns:
        LLM client instance
    """
    
    # Workflows that support knowledge base override
    kb_override_workflows = ['main_chat', 'science_traceability_matrix', 'gate_product_developer']
    
    # Check for knowledge base override first (for supported workflows)
    if workflow_name in kb_override_workflows and selected_kb:
        kb_model = None
        
        # Handle both KB name and KB object
        if isinstance(selected_kb, str) and selected_kb != "None":
            kb_model = get_model_from_knowledge_base(selected_kb)
        elif isinstance(selected_kb, dict) and selected_kb.get("base_model"):
            kb_model = selected_kb["base_model"]
        
        if kb_model:
            print(f"[WORKFLOW_MODEL] Using KB model '{kb_model}' for {workflow_name}")
            return create_model_client(kb_model, temperature)
    
    # Get default model from workflow settings
    default_model = get_default_workflow_model(workflow_name)
    print(f"[WORKFLOW_MODEL] Using default model '{default_model}' for {workflow_name}")
    return create_model_client(default_model, temperature)


def get_model_from_knowledge_base(kb_name):
    """Get the model specified by a knowledge base"""
    try:
        # Get knowledge bases from session state
        kbs = st.session_state.get("knowledge_bases", [])
        kb = next((kb for kb in kbs if kb.get("name") == kb_name), None)
        
        if kb and kb.get("base_model"):
            return kb["base_model"]
        
        return None
    except Exception as e:
        print(f"[WORKFLOW_MODEL] Error getting KB model: {e}")
        return None


def get_default_workflow_model(workflow_name):
    """Get the default model for a workflow from user preferences"""
    try:
        user_prefs = st.session_state.get("user_preferences", {})
        workflow_models = user_prefs.get("workflow_models", {})
        
        # Default fallback
        default_model = workflow_models.get(workflow_name, "azure:gpt-4o")
        return default_model
        
    except Exception as e:
        print(f"[WORKFLOW_MODEL] Error getting default model: {e}")
        return "azure:gpt-4o"


def create_model_client(model_specification, temperature=0.7):
    """
    Create a model client based on model specification.
    
    Args:
        model_specification: Either "azure:model_name" or "custom:model_name" or just "model_name"
        temperature: Temperature setting
    
    Returns:
        LLM client instance
    """
    try:
        if ":" in model_specification:
            provider, model_name = model_specification.split(":", 1)
        else:
            # Assume it's a custom model name for backward compatibility
            provider = "custom"
            model_name = model_specification
        
        if provider == "azure":
            # Use Azure OpenAI with the specified deployment
            print(f"[WORKFLOW_MODEL] Creating Azure client for {model_name}")
            return get_chat_client(temperature=temperature, deployment=model_name)
        
        elif provider == "gpt-oss":
            # Handle gpt-oss models (treat as custom endpoint)
            print(f"[WORKFLOW_MODEL] Creating gpt-oss client for {model_name}")
            # Find the gpt-oss model in user preferences
            user_prefs = st.session_state.get("user_preferences", {})
            available_models = user_prefs.get("available_models", [])
            
            # Look for a model that matches the full specification
            model_config = next((m for m in available_models if m.get("name") == model_specification), None)

            if model_config:
                return CustomEndpointClient(
                    endpoint=model_config.get("endpoint", ""),
                    api_key=model_config.get("api_key", ""),
                    model_name=model_specification,  # Use full specification
                    temperature=temperature
                )
            else:
                print(f"[WORKFLOW_MODEL] gpt-oss model {model_specification} not found, falling back to custom")
                # Try to create a custom client with the default endpoint
                return get_chat_client(temperature=temperature)

        elif provider == "custom":
            # Find the custom model in user preferences
            user_prefs = st.session_state.get("user_preferences", {})
            available_models = user_prefs.get("available_models", [])
            
            model_config = next((m for m in available_models if m.get("name") == model_name), None)
            
            if model_config:
                print(f"[WORKFLOW_MODEL] Creating custom client for {model_name}")
                return CustomEndpointClient(
                    endpoint=model_config.get("endpoint", ""),
                    api_key=model_config.get("api_key", ""),
                    model_name=model_name,
                    temperature=temperature
                )
            else:
                print(f"[WORKFLOW_MODEL] Custom model {model_name} not found, falling back to default")
                return get_chat_client(temperature=temperature)
        
        else:
            print(f"[WORKFLOW_MODEL] Unknown provider {provider}, falling back to default")
            return get_chat_client(temperature=temperature)
            
    except Exception as e:
        print(f"[WORKFLOW_MODEL] Error creating model client: {e}")
        # Fallback to default client
        return get_chat_client(temperature=temperature)


def get_workflow_model_display_name(workflow_name, selected_kb=None):
    """
    Get a human-readable display name for the model being used by a workflow.
    
    Args:
        workflow_name: Workflow identifier
        selected_kb: Selected knowledge base name (if any)
    
    Returns:
        String describing the model being used
    """
    try:
        # Workflows that support knowledge base override
        kb_override_workflows = ['main_chat', 'science_traceability_matrix', 'gate_product_developer']
        
        # Check for knowledge base override first (for supported workflows)
        if workflow_name in kb_override_workflows and selected_kb and selected_kb != "None":
            kb_model = get_model_from_knowledge_base(selected_kb)
            if kb_model:
                return f"{kb_model} (from {selected_kb} KB)"
        
        # Get default model from workflow settings
        default_model = get_default_workflow_model(workflow_name)
        
        if ":" in default_model:
            provider, model_name = default_model.split(":", 1)
            if provider == "azure":
                return f"Azure {model_name}"
            elif provider == "custom":
                return f"{model_name}"
        
        return default_model
        
    except Exception as e:
        print(f"[WORKFLOW_MODEL] Error getting display name: {e}")
        return "Azure gpt-4o (default)"
