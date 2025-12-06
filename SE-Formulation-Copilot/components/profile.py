"""
Profile component for Chat JPL application.
"""

import streamlit as st
from utils.data_persistence import save_user_profile
from components.top_nav import render_back_to_chat_button

def render_profile_page():
    """Render the profile editing page"""
    print("[Profile] Rendered UI")
    render_back_to_chat_button()
    st.markdown(
    """
    <div style="margin-top:0; padding-top:0">
        <hr style="margin-top:0; margin-bottom:0.25rem; border:1px solid #e5e7eb;" />
    </div>
    """,unsafe_allow_html=True)
    
    st.markdown('<h1 style="text-align: center;">Profile</h1>', unsafe_allow_html=True)

    # Get current profile data
    current_profile = st.session_state.get('user_profile', {
        'name': 'Kareem',
        'age': '',
        'field': '',
        'years_of_experience': ''
    })
    
    with st.form("profile_form"):        
        # Basic Information Section
        st.markdown("### Basic Information")
        new_name = st.text_input("Name", value=current_profile.get('name', ''))
        new_age = st.text_input("Age", value=str(current_profile.get('age', '')))
        
        # Professional Information Section
        st.markdown("### Professional Information")
        
        # Field text input
        current_field = current_profile.get('field', '')
        new_field = st.text_input(
            "Field/Discipline",
            value=current_field
        )
        
        new_experience = st.text_input(
            "Years of Experience",
            value=str(current_profile.get('years_of_experience', ''))
        )
        
        # Form buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.form_submit_button("Save", type="primary"):
                if new_name.strip():
                    # Validate numeric fields
                    age_valid = True
                    experience_valid = True
                    
                    if new_age and not new_age.isdigit():
                        st.error("Age must be a number")
                        age_valid = False
                    
                    if new_experience and not new_experience.replace('.', '').replace('-', '').isdigit():
                        st.error("Years of experience must be a number")
                        experience_valid = False
                    
                    if age_valid and experience_valid:
                        # Save profile
                        profile_data = {
                            'name': new_name.strip(),
                            'age': new_age.strip(),
                            'field': new_field.strip(),
                            'years_of_experience': new_experience.strip()
                        }
                        
                        st.session_state.user_profile = profile_data
                        # Save to databases folder
                        save_user_profile(st.session_state.user_profile)
                        st.success("Profile updated successfully!")
                        st.session_state.current_page = "chat"
                        st.query_params.page = "chat"
                        st.rerun()
                else:
                    st.error("Please enter a valid name")
        
    # CSS to remove form border
    st.markdown("""
        <style>
        /* Remove border from profile form */
        .stForm {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        
        /* Remove form container border */
        div[data-testid="stForm"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)