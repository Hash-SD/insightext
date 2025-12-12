"""
Sidebar components with modern UI design style.
Updates: Organized hierarchy, clear navigation, and User Mode toggle.
"""

import streamlit as st
import time
from config.settings import settings

# ... (Previous helper functions for admin kept as internal helpers) ...
import hashlib

def _hash_password(password: str) -> str:
    salt = "mlops_admin_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

def _verify_admin_password(input_password: str) -> bool:
    return input_password == settings.ADMIN_PASSWORD

def _check_admin_session() -> bool:
    return st.session_state.get('admin_authenticated', False)

def _login_admin(password: str) -> bool:
    if _verify_admin_password(password):
        st.session_state['admin_authenticated'] = True
        st.session_state['admin_login_time'] = time.time()
        return True
    return False

def _logout_admin():
    st.session_state['admin_authenticated'] = False
    st.session_state.pop('admin_login_time', None)

def _check_session_timeout(timeout_minutes: int = 30) -> bool:
    login_time = st.session_state.get('admin_login_time', 0)
    if login_time == 0:
        return True
    elapsed = time.time() - login_time
    return elapsed > (timeout_minutes * 60)

# Model Metadata definitions (Kept)
MODEL_METADATA = {
    'v1': {
        'name': 'NB Indonesian Sentiment',
        'model_type': 'MultinomialNB + TF-IDF',
        'task': 'Sentiment Analysis',
        'language': 'Indonesian',
        'labels': ['negatif', 'netral', 'positif'],
        'accuracy': 0.6972,
        'description': 'Analisis sentimen Bahasa Indonesia (3 kelas)'
    },
    'v2': {
        'name': 'NB English Sentiment',
        'model_type': 'MultinomialNB + TF-IDF',
        'task': 'Sentiment Analysis',
        'language': 'English',
        'labels': ['negative', 'positive'],
        'accuracy': 0.8647,
        'description': 'Analisis sentimen English (binary)'
    }
}

def render_sidebar(retraining_service=None) -> str:
    """
    Render modern sidebar with separated sections.
    """
    with st.sidebar:
        # 1. Header & Branding
        st.markdown(
            f"""
            <div style="margin-bottom: 20px;">
                <h1 style="font-size: 1.8rem; margin: 0; display:flex; align-items:center; gap:10px;">
                    {settings.APP_ICON} 
                    <span style="color: #1a73e8;">Text</span>AI
                </h1>
                <p style="color: #5f6368; margin: 0; font-size: 0.9rem;">
                    Enterprise Sentiment Analysis
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # 2. User Mode Selection (Crucial for UX)
        st.markdown("### 👤 User Mode")
        mode = st.radio(
            "Select Interface Mode",
            options=["Beginner", "Expert"],
            index=0 if st.session_state.get('user_mode', 'Beginner') == 'Beginner' else 1,
            label_visibility="collapsed",
            key="user_mode_radio",
            horizontal=True
        )
        st.session_state['user_mode'] = mode
        
        if mode == "Beginner":
            st.caption("Mode sederhana untuk analisis cepat.")
        else:
            st.caption("Mode detail dengan metrik performa lengkap.")
            
        st.divider()

        # 3. Main Navigation
        st.markdown("### 🧭 Navigation")
        # Use a more visual style using standard radio for now, custom CSS handles the rest
        selected_page = st.radio(
            "Go to",
            options=["🔮 Prediksi", "📊 Monitoring", "🚀 Model Management"],
            label_visibility="collapsed"
        )
        
        st.divider()

        # 4. Configuration Section
        with st.expander("⚙️  Pengaturan & Model", expanded=True):
            # Model Version
            st.markdown("**Model Version**")
            model_options = {'v1': '🇮🇩 Indonesian', 'v2': '🇺🇸 English'}
            current_version = st.session_state.get('selected_model_version', 'v1')
            
            selected_v = st.selectbox(
                "Model",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=list(model_options.keys()).index(current_version),
                label_visibility="collapsed"
            )
            st.session_state['selected_model_version'] = selected_v
            
            # Show small model info
            if mode == "Expert":
                meta = MODEL_METADATA.get(selected_v, {})
                st.info(f"{meta.get('model_type')}\nAcc: {meta.get('accuracy'):.1%}")

            # Data Consent
            st.markdown("**Data Privacy**")
            dont_save = st.checkbox(
                "Don't save my data",
                value=st.session_state.get('dont_save_data', False)
            )
            st.session_state['dont_save_data'] = dont_save
            st.session_state['user_consent'] = not dont_save

        # 5. Admin Actions (Only in Expert Mode or separate?)
        # Let's keep it accessible but unobtrusive
        if selected_page == "🚀 Model Management":
             # Logic handled in main page, but maybe login here if needed? 
             # For now, keep simple.
             pass

        # Footer
        st.markdown(
            """
            <div style="position: fixed; bottom: 20px; font-size: 0.8rem; color: #9aa0a6;">
                v2.1.0 • Built with Streamlit
            </div>
            """,
            unsafe_allow_html=True
        )
        
    return selected_page

# ... (Retraining Button function logic can be imported/kept if needed, 
# but for main UI refactor, I'm focusing on the structure provided above)
