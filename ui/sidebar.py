"""
Sidebar components untuk MLOps Streamlit Text AI application.

Module ini menyediakan fungsi untuk render sidebar dengan:
- Consent checkbox untuk data storage
- Model selector (v1: Indonesian, v2: English)
- Model info display (Naive Bayes)
- Retraining button (dengan autentikasi admin)
"""

import streamlit as st
import hashlib
import time
from typing import Optional
from config.settings import settings


def _hash_password(password: str) -> str:
    """Hash password dengan salt untuk keamanan."""
    salt = "mlops_admin_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def _verify_admin_password(input_password: str) -> bool:
    """Verifikasi password admin."""
    return input_password == settings.ADMIN_PASSWORD


def _check_admin_session() -> bool:
    """Check apakah admin sudah login dalam session."""
    return st.session_state.get('admin_authenticated', False)


def _login_admin(password: str) -> bool:
    """Login admin dengan password."""
    if _verify_admin_password(password):
        st.session_state['admin_authenticated'] = True
        st.session_state['admin_login_time'] = time.time()
        return True
    return False


def _logout_admin():
    """Logout admin."""
    st.session_state['admin_authenticated'] = False
    st.session_state.pop('admin_login_time', None)


def _check_session_timeout(timeout_minutes: int = 30) -> bool:
    """Check apakah session sudah timeout."""
    login_time = st.session_state.get('admin_login_time', 0)
    if login_time == 0:
        return True
    elapsed = time.time() - login_time
    return elapsed > (timeout_minutes * 60)


# Naive Bayes Model Metadata
MODEL_METADATA = {
    'v1': {
        'name': 'NB Indonesian Sentiment',
        'model_type': 'MultinomialNB + TF-IDF',
        'task': 'Sentiment Analysis',
        'language': 'Indonesian',
        'labels': ['negatif', 'netral', 'positif'],
        'accuracy': 0.6972,
        'f1_score': 0.6782,
        'description': 'Analisis sentimen Bahasa Indonesia (3 kelas)'
    },
    'v2': {
        'name': 'NB English Sentiment',
        'model_type': 'MultinomialNB + TF-IDF',
        'task': 'Sentiment Analysis',
        'language': 'English',
        'labels': ['negative', 'positive'],
        'accuracy': 0.8647,
        'f1_score': 0.8647,
        'description': 'Analisis sentimen English (binary)'
    }
}


def render_consent_checkbox() -> bool:
    """
    Render consent checkbox dengan tooltip explanation.
    
    Returns:
        bool: User consent status (True jika user setuju untuk simpan data)
    """
    # Checkbox untuk OPT-OUT (jangan simpan)
    dont_save = st.checkbox(
        "🚫 Jangan simpan data saya",
        value=st.session_state.get('dont_save_data', False),
        help=(
            "Centang ini jika Anda TIDAK ingin data disimpan. "
            "Secara default, data akan tersimpan untuk meningkatkan model."
        ),
        key='dont_save_checkbox'
    )
    
    # Update session state: consent = True jika TIDAK dicentang
    consent = not dont_save
    st.session_state['user_consent'] = consent
    st.session_state['dont_save_data'] = dont_save
    
    return consent


def _render_model_selection_ui() -> str:
    """
    Display model selector UI.
    
    Returns:
        str: Model version ('v1' atau 'v2')
    """
    # Model options
    model_options = {
        'v1': '🇮🇩 Indonesian',
        'v2': '🇺🇸 English (Default)'
    }
    
    # Get current selection from session state
    current_version = st.session_state.get('selected_model_version', 'v1')
    
    # Model selector
    selected = st.selectbox(
        "Pilih Versi Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(current_version),
        help="Pilih model yang sesuai dengan bahasa teks Anda."
    )
    
    st.session_state['selected_model_version'] = selected
    return selected


def _display_model_details(version: str):
    """
    Display model details inside an expander.
    
    Args:
        version: Model version (e.g., 'v1')
    """
    metadata = MODEL_METADATA.get(version, {})
    
    if metadata:
        with st.expander("🔬 Detail Model Aktif", expanded=False):
            st.markdown(f"**🧠 {metadata.get('name', 'Unknown Model')}**")
            st.caption(f"{metadata.get('model_type')} | {metadata.get('language')}")
            
            # Display labels
            labels = metadata.get('labels', [])
            st.markdown(f"**Output:** {', '.join(labels)}")
            
            # Description
            st.markdown(f"_{metadata.get('description', '')}_")


def render_retrain_button() -> bool:
    """
    Render retraining button dengan confirmation dialog.
    
    Returns:
        bool: True jika user mengkonfirmasi retraining
    """
    st.markdown("### 🔄 Retraining Pipeline")
    
    # Button untuk trigger retraining
    if st.button(
        "Latih Ulang Model",
        help="Trigger retraining pipeline dengan data terbaru",
        width="stretch",
        type="primary"
    ):
        # Set flag untuk show confirmation dialog
        st.session_state['show_retrain_confirmation'] = True
    
    # Show confirmation dialog jika flag is set
    if st.session_state.get('show_retrain_confirmation', False):
        st.warning(
            "⚠️ **Konfirmasi Retraining**\n\n"
            "Proses ini akan melatih model baru dengan data terbaru.\n"
            "Apakah Anda yakin?"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Ya", width="stretch"):
                st.session_state['show_retrain_confirmation'] = False
                st.session_state['confirm_retrain'] = True
                st.rerun()
        
        with col2:
            if st.button("❌ Batal", width="stretch"):
                st.session_state['show_retrain_confirmation'] = False
                st.session_state['confirm_retrain'] = False
                st.rerun()
    
    # Return confirmation status
    confirmed = st.session_state.get('confirm_retrain', False)
    
    # Reset confirmation flag after returning
    if confirmed:
        st.session_state['confirm_retrain'] = False
    
    return confirmed


def render_sidebar(retraining_service=None) -> str:
    """
    Main function untuk render complete sidebar.
    
    Orchestrates semua sidebar components dengan layout baru.
    
    Returns:
        str: Selected page ('Prediksi' or 'Monitoring')
    """
    with st.sidebar:
        # App title - diperbesar dengan HTML
        st.markdown(
            f"<h1 style='text-align: left; font-size: 2.5rem; margin-bottom: 0;'>{settings.APP_ICON} {settings.APP_TITLE}</h1>",
            unsafe_allow_html=True
        )
        
        st.divider()
        
        # Navigation
        st.header("🧭 Navigasi")
        page = st.radio(
            "Pilih Halaman:",
            ["🔮 Prediksi", "📊 Monitoring", "🚀 Model Management"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # 1. Pengaturan Model (Fokus Utama)
        st.header("⚙️ Pengaturan Model")
        
        # Model selector
        selected_version = _render_model_selection_ui()
        
        # Divider halus
        st.divider()
        
        # Detail Model (Hidden by default)
        _display_model_details(selected_version)
        
        st.divider()
        
        # 2. Pengaturan Lain
        st.subheader("🛠️ Pengaturan Lain")
        render_consent_checkbox()
        
        st.divider()
        
        # 3. Akses Admin (Footer)
        with st.expander("👤 Akses Admin"):
            if retraining_service:
                _render_admin_section(retraining_service)
            else:
                st.info("Service retraining tidak tersedia.")
        
        # Footer Version
        st.caption("Kelompok 6 UHUY😎😋 | © 2025")
        
    return page


def _render_admin_section(retraining_service):
    """
    Render admin section dengan autentikasi password.
    
    Args:
        retraining_service: RetrainingService instance
    """
    # Check session timeout
    if _check_admin_session() and _check_session_timeout(timeout_minutes=30):
        _logout_admin()
        st.warning("⏰ Session timeout.")
    
    # Check if admin is authenticated
    if not _check_admin_session():
        # Login form
        password = st.text_input(
            "Password Admin", 
            type="password",
            placeholder="Masukkan password",
            key="admin_pass_input"
        )
        
        if st.button("🔓 Masuk", use_container_width=True):
            if password:
                if _login_admin(password):
                    st.success("Login berhasil!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Password salah!")
                    time.sleep(1)
            else:
                st.error("Masukkan password!")
    else:
        # Admin authenticated
        st.success("✅ Admin Logged In")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            _logout_admin()
            st.rerun()
        
        st.markdown("---")
        
        # Retraining section
        retrain_confirmed = render_retrain_button()
        
        # Handle retraining if confirmed
        if retrain_confirmed:
            _handle_retraining(retraining_service)


def _handle_retraining(retraining_service):
    """
    Handle retraining flow dengan spinner, progress bar, dan error handling.
    
    Args:
        retraining_service: RetrainingService instance
    """
    from utils.logger import setup_logger, log_error
    import logging
    import time
    
    logger = setup_logger('retraining', 'app.log', logging.INFO)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Get selected model version
        model_version = st.session_state.get('selected_model_version', 'v1')
        
        # Step 1: Validate requirements
        status_text.text("⏳ Memvalidasi requirements...")
        progress_bar.progress(10)
        time.sleep(0.3)
        
        is_valid, validation_msg = retraining_service.validate_retraining_requirements()
        
        if not is_valid:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Validasi gagal: {validation_msg}")
            logger.warning(f"Retraining validation failed: {validation_msg}")
            return
        
        # Step 2: Fetch dataset
        status_text.text("⏳ Mengambil dataset snapshot...")
        progress_bar.progress(30)
        time.sleep(0.3)
        
        # Step 3: Training model
        status_text.text("⏳ Melatih model baru...")
        progress_bar.progress(50)
        
        # Trigger retraining
        result = retraining_service.trigger_retraining(model_version=model_version)
        
        # Step 4: Evaluating
        status_text.text("⏳ Mengevaluasi model...")
        progress_bar.progress(80)
        time.sleep(0.3)
        
        # Step 5: Complete
        progress_bar.progress(100)
        status_text.text("✅ Selesai!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Check status
        if result.get('status') == 'success':
            st.success(
                f"✅ Retraining berhasil!\n\n"
                f"Model baru: {result.get('new_version', 'N/A')}\n\n"
                f"Metrik:\n"
                f"- Akurasi: {result.get('metrics', {}).get('accuracy', 0):.2%}\n"
                f"- F1 Score: {result.get('metrics', {}).get('f1_score', 0):.2%}"
            )
            st.toast("✅ Model berhasil dilatih ulang!", icon="🎉")
            logger.info(f"Retraining completed successfully: {result}")
        elif result.get('status') == 'no_data':
            st.warning(
                f"⚠️ {result.get('message', 'Tidak ada data untuk retraining')}\n\n"
                "Pastikan ada prediksi dengan user consent yang diaktifkan."
            )
            logger.warning(f"Retraining skipped - no data: {result}")
        else:
            st.error(
                f"❌ Retraining gagal: {result.get('message', 'Kesalahan tidak diketahui')}"
            )
            logger.error(f"Retraining failed: {result}")
            
    except ValueError as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Parameter tidak valid: {str(e)}")
        logger.error(f"Retraining validation error: {e}")
        log_error(logger, e, {'model_version': model_version})
    except ConnectionError as e:
        progress_bar.empty()
        status_text.empty()
        st.error(
            f"❌ Gagal terhubung ke database atau MLflow:\n\n{str(e)}\n\n"
            "Periksa koneksi dan coba lagi."
        )
        logger.error(f"Retraining connection error: {e}")
        log_error(logger, e, {'model_version': model_version})
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(
            f"❌ Terjadi kesalahan saat retraining:\n\n{str(e)}\n\n"
            "Silakan coba lagi atau hubungi administrator."
        )
        logger.error(f"Retraining failed: {e}")
        log_error(logger, e, {'model_version': model_version})
