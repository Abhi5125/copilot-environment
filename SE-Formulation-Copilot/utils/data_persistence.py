"""
Data persistence utilities for Chat JPL application.
Handles conversations, user profiles, and other data storage.
"""

import json
import os
import uuid
import streamlit as st

def load_conversations():
    """Load conversations from databases folder"""
    try:
        if os.path.exists("databases/conversations.json"):
            with open("databases/conversations.json", "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return []

def save_conversations(conversations):
    """Save conversations to databases folder"""
    try:
        os.makedirs("databases", exist_ok=True)
        with open("databases/conversations.json", "w") as f:
            json.dump(conversations, f, indent=2)
    except Exception as e:
        print(f"Error saving conversations: {e}")

def load_user_profile():
    """Load user profile from user_preferences.json"""
    try:
        # Load user preferences which now contains profile information
        user_preferences = load_user_preferences()
        
        # Extract the user_profile section
        user_profile_section = user_preferences.get("user_profile", {})
        
        # Return the profile data
        profile = {
            "name": user_profile_section.get("name", "User"),
            "age": user_profile_section.get("age", ""),
            "field": user_profile_section.get("field", ""),
            "years_of_experience": user_profile_section.get("years_of_experience", "")
        }
        
        return profile
    except Exception as e:
        print(f"Error loading user profile: {e}")
        return {"name": "User"}

def save_user_profile(user_profile):
    """Save user profile to user_preferences.json"""
    try:
        # Load existing preferences
        user_preferences = load_user_preferences()
        
        # Ensure user_profile section exists
        if "user_profile" not in user_preferences:
            user_preferences["user_profile"] = {}
        
        # Update profile fields in the user_profile section
        user_preferences["user_profile"]["name"] = user_profile.get("name", "User")
        if "age" in user_profile:
            user_preferences["user_profile"]["age"] = user_profile["age"]
        if "field" in user_profile:
            user_preferences["user_profile"]["field"] = user_profile["field"]
        if "years_of_experience" in user_profile:
            user_preferences["user_profile"]["years_of_experience"] = user_profile["years_of_experience"]
        
        # Save updated preferences
        save_user_preferences(user_preferences)
    except Exception as e:
        print(f"Error saving user profile: {e}")

def load_knowledge_bases():
    """Load knowledge bases from unified databases folder - for backward compatibility"""
    return load_custom_knowledge_bases()

def load_custom_knowledge_bases():
    """Load knowledge bases from user_preferences.json"""
    try:
        user_preferences = load_user_preferences()
        kbs = user_preferences.get('custom_knowledge_bases', [])
        
        # Add IDs to existing KBs that don't have them (migration)
        updated = False
        for kb in kbs:
            if 'id' not in kb:
                kb['id'] = str(uuid.uuid4())[:8]  # 8-character unique ID
                updated = True
                print(f"[MIGRATION] Added ID '{kb['id']}' to KB '{kb['name']}'")
        
        if updated:
            user_preferences['custom_knowledge_bases'] = kbs
            save_user_preferences(user_preferences)
            print(f"[MIGRATION] Updated {len([kb for kb in kbs if 'id' in kb])} KBs with unique IDs")
        
        return kbs
    except Exception as e:
        print(f"Error loading knowledge bases: {e}")
        return []

def generate_kb_id():
    """Generate a unique 8-character ID for a knowledge base"""
    return str(uuid.uuid4())[:8]

def save_knowledge_bases(knowledge_bases):
    """Save knowledge bases to unified system - for backward compatibility"""
    save_custom_knowledge_bases(knowledge_bases)

def save_custom_knowledge_bases(knowledge_bases):
    """Save knowledge bases to user_preferences.json"""
    try:
        # Remove any source fields if they exist (cleanup)
        clean_kbs = []
        for kb in knowledge_bases:
            kb_copy = kb.copy()
            kb_copy.pop("source", None)  # Remove source field if it exists
            clean_kbs.append(kb_copy)
        
        # Update user preferences
        user_preferences = load_user_preferences()
        user_preferences['custom_knowledge_bases'] = clean_kbs
        save_user_preferences(user_preferences)
    except Exception as e:
        print(f"Error saving knowledge bases: {e}")

def load_rag_settings():
    """Load RAG settings from user_preferences.json"""
    try:
        user_preferences = load_user_preferences()
        rag_settings = user_preferences.get("rag_settings", {})
        
        # Return RAG settings with defaults if not present
        return {
            "full_context_mode": rag_settings.get("full_context_mode", False),
            "hybrid_search": rag_settings.get("hybrid_search", True),
            "reranking_engine": rag_settings.get("reranking_engine", False),
            "reranking_model": rag_settings.get("reranking_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            "top_k": rag_settings.get("top_k", 5),
            "top_k_reranker": rag_settings.get("top_k_reranker", 3),
            "relevance_threshold": rag_settings.get("relevance_threshold", 0.5),
            "bm25_weight": rag_settings.get("bm25_weight", 0.5),
            "query_expansion": rag_settings.get("query_expansion", True),
            "num_query_variations": rag_settings.get("num_query_variations", 3)
        }
    except Exception as e:
        print(f"Error loading RAG settings: {e}")
        return {
            "full_context_mode": False,
            "hybrid_search": True,
            "reranking_engine": False,
            "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": 5,
            "top_k_reranker": 3,
            "relevance_threshold": 0.5,
            "bm25_weight": 0.5,
            "query_expansion": True,
            "num_query_variations": 3
        }

def save_rag_settings(rag_settings):
    """Save RAG settings to user_preferences.json"""
    try:
        user_preferences = load_user_preferences()
        user_preferences["rag_settings"] = rag_settings
        save_user_preferences(user_preferences)
    except Exception as e:
        print(f"Error saving RAG settings: {e}")

def load_user_preferences():
    """Load user preferences from databases folder"""
    try:
        if os.path.exists("databases/user_preferences.json"):
            with open("databases/user_preferences.json", "r") as f:
                return json.load(f)
        return {"selected_knowledge_base": "None"}
    except Exception as e:
        print(f"Error loading user preferences: {e}")
        return {"selected_knowledge_base": "None"}

def save_user_preferences(user_preferences):
    """Save user preferences to databases folder"""
    try:
        os.makedirs("databases", exist_ok=True)
        with open("databases/user_preferences.json", "w") as f:
            json.dump(user_preferences, f, indent=2)
    except Exception as e:
        print(f"Error saving user preferences: {e}")

def add_knowledge_base(kb_data):
    """Add a new knowledge base to the unified system"""
    try:
        # Mark as custom by default
        kb_data["source"] = "custom"
        
        # Add to session state
        if "knowledge_bases" not in st.session_state:
            st.session_state.knowledge_bases = []
        
        st.session_state.knowledge_bases.append(kb_data)
        
        # Save to file
        save_knowledge_bases(st.session_state.knowledge_bases)
        
        return True
    except Exception as e:
        print(f"Error adding knowledge base: {e}")
        return False

def get_selected_knowledge_base():
    """Get the currently selected knowledge base"""
    try:
        preferences = load_user_preferences()
        selected_kb_name = preferences.get("selected_knowledge_base", "None")
        
        if selected_kb_name == "None":
            return None
            
        # Load all knowledge bases from unified file
        all_kbs = load_custom_knowledge_bases()
        
        # Find the selected knowledge base
        for kb in all_kbs:
            if kb.get("name") == selected_kb_name:
                return kb
                
        return None
    except Exception as e:
        print(f"Error getting selected knowledge base: {e}")
        return None

def set_selected_knowledge_base(kb_name):
    """Set the selected knowledge base"""
    try:
        preferences = load_user_preferences()
        preferences["selected_knowledge_base"] = kb_name
        save_user_preferences(preferences)
        
        # Also update session state if it exists
        if "user_preferences" in st.session_state:
            st.session_state.user_preferences = preferences
            
        return True
    except Exception as e:
        print(f"Error setting selected knowledge base: {e}")
        return False