import streamlit as st
from openai import OpenAI
import requests
import yaml

# Session state setup
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None

# Initialize OpenRouter client
client = OpenAI(
    api_key='sMbquLrxy0DnjpZDhVdfZYCwVe19jkSK',
    base_url="https://api.mistral.ai/v1"
)

def fetch_settings(owner, repo, ref='main', pat=''):
    if ref == 'unversioned':
        ref = 'main'
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/.settings.yaml"
    headers = {}
    if pat:
        headers["Authorization"] = f"token {pat}"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return {}  # No .settings.yaml, skip

    return yaml.safe_load(r.text)

def get_ref_sha(owner, repo, ref, pat=''):
    if ref == 'unversioned' or ref == 'main':
        ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/main"
    else:
        ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/tags/{ref}"
    
    headers = {"Accept": "application/vnd.github+json"}
    if pat:
        headers["Authorization"] = f"token {pat}"
    r = requests.get(ref_url, headers=headers)
    if r.status_code != 200:
        raise ValueError(f"No ref found for {ref} (status: {r.status_code})")
    
    return r.json()['object']['sha']

def get_repo_tree(owner, repo, sha, pat=''):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
    headers = {"Accept": "application/vnd.github+json"}
    if pat:
        headers["Authorization"] = f"token {pat}"

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise ValueError("Failed to get tree (status: {r.status_code})")
    
    return r.json().get('tree', [])

def fetch_knowledge(owner, repo, version, visited=None, pat=''):
    if visited is None:
        visited = set()
    
    if repo in visited:
        return ""
    
    visited.add(repo)
    
    if version == 'unversioned':
        version = 'main'
    
    sha = get_ref_sha(owner, repo, version, pat)
    tree = get_repo_tree(owner, repo, sha, pat)
    
    knowledge = f"Repo: {repo} version: {version}\n"
    
    for item in tree:
        if item['type'] == 'blob':
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{version}/{item['path']}"
            r = requests.get(url)
            if r.status_code == 200:
                try:
                    text = r.text
                    knowledge += f"\nFile: {repo}/{item['path']}\n{text}\n"
                except:
                    pass  # Skip non-text/binary files
    
    # Get dependencies
    settings = fetch_settings(owner, repo, version, pat)
    deps = settings.get('dependencies', {})
    
    if isinstance(deps, list):
        deps = {dep: "unversioned" for dep in deps}
    
    for dep_repo, dep_version in deps.items():
        if dep_version in ['0.0.0', 'unspecified']:
            dep_version = 'main'
        try:
            knowledge += fetch_knowledge(owner, dep_repo, dep_version, visited, pat)
        except ValueError:
            pass  # Skip invalid deps
    
    return knowledge

# Main UI
repos = ["project-environment", "fire-warden", "mission-control", "modeling-environment", "copilot-environment", "fire-cloud"]

st.sidebar.title("Select Knowledge Base")
selected_repo = st.sidebar.selectbox("Select Repo", options=repos)
version = st.sidebar.text_input("Enter Version (e.g., v1.0.0)", value="main")
pat = st.sidebar.text_input("GitHub PAT (optional for rate limits)", type="password")
load_button = st.sidebar.button("Load Knowledge Base")

if load_button:
    st.write(f"Attempting to load version: {version}")  # Debug print to verify input
    try:
        owner = "fireforce6-f25"
        kb = fetch_knowledge(owner, selected_repo, version, pat=pat)
        st.session_state.knowledge_base = kb
        st.sidebar.success("Knowledge base loaded from repo and dependencies!")
    except Exception as e:
        st.sidebar.error(f"Error loading: {str(e)}")

# Chat interface if knowledge base is loaded
if st.session_state.knowledge_base:
    st.title("Chat Interface")
    user_query = st.text_input("Enter your query:")
    if user_query:
        # Build prompt with knowledge base context
        context = st.session_state.knowledge_base[:64000]  # Increased for full coverage
        prompt = f"Based on this knowledge base: '{context}'\nAnswer the query: {user_query}"
        
        # Call OpenRouter API
        try:
            response = client.chat.completions.create(
                model="mistral-small-latest",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}]
            ).choices[0].message.content
        except Exception as e:
            response = f"Error calling AI: {str(e)}"
        
        st.session_state.chat_history.append({"user": user_query, "bot": response})
    
    # Display chat history
    for msg in st.session_state.chat_history:
        st.write(f"User: {msg['user']}")
        st.write(f"Bot: {msg['bot']}")
else:
    st.info("Load a knowledge base to start chatting.")