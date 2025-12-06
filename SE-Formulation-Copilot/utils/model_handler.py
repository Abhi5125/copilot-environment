"""
Centralized Model Handler
Handles all LLM operations for the SE Copilot application.
Supports Claude and GPT-OSS models via custom endpoints.
"""

import os
import json
import requests
import urllib3
import streamlit as st
from typing import Optional, Dict, Any, List, Union
from utils.data_persistence import load_user_preferences

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Try to import Azure dependencies for embeddings only
try:
    from azure.identity import ClientSecretCredential, get_bearer_token_provider
    from langchain_openai import AzureOpenAIEmbeddings
    from dotenv import load_dotenv, find_dotenv
    AZURE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    AZURE_EMBEDDINGS_AVAILABLE = False
    print("Azure embedding dependencies not available")

# Try to import LangChain for consistency
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available - using direct API calls")
    # Define a dummy BaseMessage class for type hints when LangChain is not available
    class BaseMessage:
        def __init__(self, content: str):
            self.content = content


class ModelResponse:
    """Response object to maintain compatibility with LangChain interfaces"""
    def __init__(self, content: str):
        self.content = content


class BaseModelClient:
    """Base class for all model clients"""
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
    
    def invoke(self, messages: List[Union[BaseMessage, Dict[str, str]]]) -> ModelResponse:
        """Invoke the model with messages"""
        raise NotImplementedError("Subclasses must implement invoke method")
    
    def _convert_messages_to_api_format(self, messages: List[Union[BaseMessage, Dict[str, str]]]) -> List[Dict[str, Any]]:
        """Convert various message formats to standard API format"""
        api_messages = []
        for msg in messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # LangChain message
                if msg.type == 'system':
                    api_messages.append({"role": "system", "content": msg.content})
                elif msg.type == 'human':
                    api_messages.append({"role": "user", "content": msg.content})
                elif msg.type == 'ai':
                    api_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, dict):
                # Already in API format - preserve complex content structures for vision
                api_messages.append(msg)
            else:
                # String message - assume user
                api_messages.append({"role": "user", "content": str(msg)})
        return api_messages


class CustomEndpointClient(BaseModelClient):
    """Client for custom API endpoints (GPT-OSS, Bedrock, etc.)"""
    
    def __init__(self, endpoint: str, api_key: str, model_name: str, 
                 temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.endpoint = endpoint
        self.api_key = api_key
    
    def _map_model_name_for_api(self, model_name: str) -> str:
        """Map internal model names to actual API model names"""
        model_mapping = {
            # Only map models that don't exist in the API
            "gpt-5-nano": "gpt-4o-mini"  # Map gpt-5-nano to gpt-4o-mini
        }
        return model_mapping.get(model_name, model_name)
    
    def invoke(self, messages: List[Union[BaseMessage, Dict[str, str]]]) -> ModelResponse:
        """Invoke the custom endpoint with messages"""
        try:
            api_messages = self._convert_messages_to_api_format(messages)
            
            # Map the model name for the API call
            api_model_name = self._map_model_name_for_api(self.model_name)
            
            payload = {
                "model": api_model_name,
                "messages": api_messages
            }
            
            # GPT-5 models use max_completion_tokens instead of max_tokens
            if api_model_name.startswith('gpt-5'):
                payload["max_completion_tokens"] = 2000
            else:
                payload["max_tokens"] = 2000
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
            }
            
            # Remove empty Authorization header
            if not self.api_key:
                headers.pop("Authorization", None)
            
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                verify=False,  # For self-signed certificates
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return ModelResponse(content)
            else:
                error_msg = f"API error {response.status_code}: {response.text}"
                print(f"[MODEL_HANDLER] {error_msg}")
                return ModelResponse(f"Error: {error_msg}")
                
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"[MODEL_HANDLER] {error_msg}")
            return ModelResponse(f"Error: {error_msg}")


class EmbeddingsClient:
    """Client for generating embeddings using OpenAI or Azure OpenAI"""
    
    def __init__(self, provider: str = "azure", model_name: str = None, api_key: str = None, endpoint: str = None):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = endpoint
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate embeddings client"""
        if self.provider == "openai":
            # For OpenAI API, we'll use direct API calls
            return None  # We'll handle embeddings in embed_documents method
        elif self.provider == "azure" and AZURE_EMBEDDINGS_AVAILABLE:
            return self._initialize_azure_embeddings()
        else:
            raise NotImplementedError(f"Embeddings provider '{self.provider}' not available")
    
    def _initialize_azure_embeddings(self):
        """Initialize Azure OpenAI embeddings client for text-embedding-3"""
        load_dotenv(find_dotenv(usecwd=True), override=True)
        
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET") 
        api_base = os.getenv("AZURE_API_BASE")
        api_version = os.getenv("AZURE_API_VERSION", "2024-07-01-preview")
        embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_ID", "text-embedding-3-large")
        
        if not all([tenant_id, client_id, client_secret, api_base, embedding_deployment]):
            raise ValueError("Missing Azure embeddings configuration")
        
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default"
        )
        
        default_headers = {}
        apim_key = os.getenv("APIM_SUBSCRIPTION_KEY")
        if apim_key:
            default_headers["Ocp-Apim-Subscription-Key"] = apim_key
        
        return AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
            api_version=api_version,
            azure_endpoint=api_base,
            azure_ad_token_provider=token_provider,
            default_headers=default_headers
        )
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts"""
        try:
            if self.provider == "openai":
                # Use OpenAI API directly
                return self._openai_embed_documents(texts)
            else:
                # Use Azure embeddings
                return self.client.embed_documents(texts)
        except Exception as e:
            print(f"[MODEL_HANDLER] Embeddings error: {e}")
            raise
    
    def _openai_embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            verify=False,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
        else:
            raise Exception(f"OpenAI embeddings API error: HTTP {response.status_code} - {response.text}")



class ModelHandler:
    """Centralized model handler for all LLM operations"""
    
    def __init__(self):
        self.chat_clients = {}
        self.embeddings_client = None
    
    def get_chat_client(self, workflow_name: str = "main_chat", temperature: float = 0.7, 
                       selected_kb: Optional[str] = None) -> BaseModelClient:
        """
        Get the appropriate chat client for a workflow.
        
        Args:
            workflow_name: The workflow identifier
            temperature: Temperature setting for the model
            selected_kb: Selected knowledge base name (for KB-specific model overrides)
        
        Returns:
            BaseModelClient instance
        """
        # Get model configuration from preferences and KB settings
        model_config = self._get_model_config(workflow_name, selected_kb)
        
        # Create cache key
        cache_key = f"{workflow_name}_{model_config['name']}_{temperature}_{selected_kb or 'none'}"
        
        # Return cached client if available
        if cache_key in self.chat_clients:
            return self.chat_clients[cache_key]
        
        # Create new client
        client = self._create_client(model_config, temperature)
        self.chat_clients[cache_key] = client
        
        return client
    
    def get_embeddings_client(self, kb_name: Optional[str] = None) -> EmbeddingsClient:
        """
        Get the embeddings client for the specified knowledge base.
        Uses the embedding model configured in the KB settings.
        
        Args:
            kb_name: Knowledge base name to get embedding model from
            
        Returns:
            EmbeddingsClient instance configured for the KB's embedding model
        """
        # Get embedding model from KB configuration
        embedding_model_config = None
        
        if kb_name and kb_name != "None":
            try:
                # Get KB list from session state
                knowledge_bases = st.session_state.get("knowledge_bases", [])
                
                # Find the KB
                kb = next((kb for kb in knowledge_bases if kb.get("name") == kb_name), None)
                if kb and kb.get("embedding_model"):
                    embedding_model_name = kb["embedding_model"]
                    # Get the model config from available models
                    embedding_model_config = self._find_model_by_name(embedding_model_name)
            except Exception as e:
                print(f"[MODEL_HANDLER] Error getting KB embedding model: {e}")
        
        # If no KB-specific embedding model, try to get a default one
        if not embedding_model_config:
            # Get first available embedding model from user preferences
            user_prefs = load_user_preferences()
            available_models = user_prefs.get("available_models", [])
            embedding_models = [m for m in available_models if m.get("type") == "embedding"]
            
            if embedding_models:
                embedding_model_config = embedding_models[0]
            else:
                # Fall back to Azure if configured
                if self.embeddings_client is None:
                    self.embeddings_client = EmbeddingsClient(provider="azure")
                return self.embeddings_client
        
        # Create a new embeddings client with the configured model
        return EmbeddingsClient(
            provider="openai",
            model_name=embedding_model_config["name"],
            api_key=embedding_model_config["api_key"],
            endpoint=embedding_model_config["endpoint"]
        )
    
    def _get_model_config(self, workflow_name: str, selected_kb: Optional[str] = None) -> Dict[str, Any]:
        """Get model configuration for a workflow"""
        
        # Check for KB-specific model override first
        if selected_kb and selected_kb != "None":
            kb_model = self._get_model_from_knowledge_base(selected_kb)
            if kb_model:
                return kb_model
        
        # Get default model from workflow preferences
        default_model = self._get_default_workflow_model(workflow_name)
        return default_model
    
    def _get_model_from_knowledge_base(self, kb_name: str) -> Optional[Dict[str, Any]]:
        """Get model configuration from knowledge base settings"""
        try:
            # Get KB list from session state
            knowledge_bases = st.session_state.get("knowledge_bases", [])
            
            # Find the KB
            kb = next((kb for kb in knowledge_bases if kb.get("name") == kb_name), None)
            if not kb or not kb.get("base_model"):
                return None
            
            base_model_name = kb["base_model"]
            
            # Get the model config from available models
            return self._find_model_by_name(base_model_name)
            
        except Exception as e:
            print(f"[MODEL_HANDLER] Error getting KB model: {e}")
            return None
    
    def _get_default_workflow_model(self, workflow_name: str) -> Dict[str, Any]:
        """Get default model configuration for a workflow"""
        try:
            # Map workflow names to model preference keys (some workflows share model settings)
            workflow_model_mapping = {
                'main_chat': 'main_chat',
                'science_traceability_matrix': 'science_traceability_matrix',
                'gate_product_developer': 'gate_product_developer', 
                'proposal_assistant': 'proposal_assistant',
                'proposal_writing': 'proposal_assistant',  # Maps to proposal_assistant model
                'ao_comparison': 'proposal_assistant',     # Maps to proposal_assistant model
                'heritage_finder': 'heritage_finder'
            }
            
            # Get the model preference key for this workflow
            model_pref_key = workflow_model_mapping.get(workflow_name, workflow_name)
            print(f"[MODEL_HANDLER] Workflow '{workflow_name}' mapped to model key '{model_pref_key}'")
            
            # Load user preferences
            user_prefs = load_user_preferences()
            workflow_models = user_prefs.get("workflow_models", {})
            print(f"[MODEL_HANDLER] Available workflow models: {list(workflow_models.keys())}")
            
            # Get the model name for this workflow using the mapped key
            model_name = workflow_models.get(model_pref_key)
            print(f"[MODEL_HANDLER] Selected model for '{model_pref_key}': {model_name}")
            
            if not model_name:
                # Fallback to first available model
                available_models = user_prefs.get("available_models", [])
                if available_models:
                    return available_models[0]
                else:
                    raise ValueError("No available models configured")
            
            # Find the model configuration - all models should be in available_models
            model_config = self._find_model_by_name(model_name)
            
            if not model_config:
                available_model_names = [m.get('name', 'unnamed') for m in user_prefs.get("available_models", [])]
                raise ValueError(f"Model '{model_name}' not found in available models: {available_model_names}")
            
            print(f"[MODEL_HANDLER] Found model config for '{model_name}': {model_config.get('name')}")
            return model_config
            
        except Exception as e:
            # Emergency fallback - use the first available model
            try:
                user_prefs = load_user_preferences()
                available_models = user_prefs.get("available_models", [])
                if available_models:
                    return available_models[0]
            except:
                pass
            
            raise ValueError(f"No fallback model available: {str(e)}")
    
    def _find_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Find model configuration by name"""
        try:
            user_prefs = load_user_preferences()
            available_models = user_prefs.get("available_models", [])
            
            return next((model for model in available_models if model.get("name") == model_name), None)
            
        except Exception as e:
            print(f"[MODEL_HANDLER] Error finding model: {e}")
            return None
    
    def _create_client(self, model_config: Dict[str, Any], temperature: float) -> BaseModelClient:
        """Create the appropriate client based on model configuration"""
        model_name = model_config.get("name", "unknown")
        
        # Only support custom endpoints (GPT-OSS, Bedrock, etc.)
        return CustomEndpointClient(
            endpoint=model_config.get("endpoint", ""),
            api_key=model_config.get("api_key", ""),
            model_name=model_name,
            temperature=temperature
        )


# Global instance
_model_handler = None

def get_model_handler() -> ModelHandler:
    """Get the global model handler instance"""
    global _model_handler
    if _model_handler is None:
        _model_handler = ModelHandler()
    return _model_handler


# Convenience functions for backward compatibility
def get_chat_client(workflow_name: str = "main_chat", temperature: float = 0.7, 
                   selected_kb: Optional[str] = None) -> BaseModelClient:
    """Get a chat client for the specified workflow"""
    return get_model_handler().get_chat_client(workflow_name, temperature, selected_kb)


def get_embeddings_client(kb_name: Optional[str] = None) -> EmbeddingsClient:
    """
    Get the embeddings client for the specified knowledge base.
    
    Args:
        kb_name: Knowledge base name to get embedding model from
        
    Returns:
        EmbeddingsClient configured for the KB's embedding model
    """
    return get_model_handler().get_embeddings_client(kb_name)


def get_workflow_kb_and_client(workflow_name: str, temperature: float = 0.7):
    """
    Get the knowledge base selection and chat client for a workflow.
    Replacement for kb_workflow_helper function.
    
    Returns:
        tuple: (selected_kb_name, selected_kb, chat_client)
    """
    try:
        # Map workflow names to preference keys
        workflow_kb_mapping = {
            'main_chat': 'main_chat_kb',
            'science_traceability_matrix': 'stm_kb',
            'gate_product_developer': 'gate_product_kb',
            'proposal_assistant': 'proposal_assistant_kb',
            'proposal_writing': 'proposal_assistant_kb',
            'ao_comparison': 'proposal_assistant_kb', 
            'heritage_finder': 'heritage_finder_kb'
        }
        
        # Get the preference key for this workflow
        pref_key = workflow_kb_mapping.get(workflow_name, 'main_chat_kb')
        
        # Get KB preferences from user preferences
        kb_prefs = st.session_state.user_preferences.get("selected_knowledge_base", {})
        selected_kb_name = kb_prefs.get(pref_key, "None")
        
        # Determine the KB to use for RAG
        selected_kb = None if selected_kb_name == "None" else selected_kb_name
        
        # Get the chat client
        chat_client = get_chat_client(workflow_name, temperature, selected_kb)
        
        return selected_kb_name, selected_kb, chat_client
        
    except Exception as e:
        # Return fallback values
        return "None", None, get_chat_client("main_chat", temperature)


def get_workflow_model_client(workflow_name: str, selected_kb: Optional[str] = None, temperature: float = 0.7):
    """
    Get the appropriate model client for a workflow tool.
    Replacement for workflow_models function.
    """
    return get_chat_client(workflow_name, temperature, selected_kb)


def get_workflow_model_display_name(workflow_name: str) -> str:
    """Get the display name for the model used by a workflow"""
    try:
        client = get_chat_client(workflow_name)
        return client.model_name
    except Exception as e:
        print(f"[MODEL_HANDLER] Error getting model display name: {e}")
        return "Unknown Model"

def get_default_model(workflow_name: str = "main_chat") -> str:
    """Get the default model name for a given workflow"""
    try:
        handler = get_model_handler()
        model_config = handler._get_default_workflow_model(workflow_name)
        if isinstance(model_config, dict):
            return model_config.get("name", "gpt-oss:120b-64k")
        return str(model_config) if model_config else "gpt-oss:120b-64k"
    except Exception as e:
        print(f"[MODEL_HANDLER] Error getting default model: {e}")
        return "gpt-oss:120b-64k"  # Fallback to a default
