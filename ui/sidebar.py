"""
Sidebar components untuk MLOps Streamlit Text AI application.

Module ini menyediakan fungsi untuk render sidebar dengan:
- Consent checkbox untuk data storage
- Model selector (v1: Indonesian, v2: IMDB English)
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
        'name': 'Naive Bayes Indonesian',
        'model_type': 'MultinomialNB + TF-IDF',
        'task': 'Sentiment Analysis',
        'language': 'Indonesian',
        'labels': ['negatif', 'netral', 'positif'],
        'accuracy': 0.6972,
        'f1_score': 0.6782,
        'description': 'Model sentiment analysis untuk teks Bahasa Indonesia (3 kelas)'
    },
    'v2': {
        'name': 'Naive Bayes IMDB',
        'model_type': 'MultinomialNB + TF-IDF',
        'task': 'Sentiment Analysis',
        'language': 'English',
        'labels': ['negative', 'positive'],
        'accuracy': 0.8647,
        'f1_score': 0.8647,
        'description': 'Model sentiment analysis untuk teks English (binary: positive/negative)'
    }
}


def render_consent_checkbox() -> bool:
    """
    Render consent checkbox dengan tooltip explanation.
    
    Returns:
        bool: User consent status (True jika user setuju)
    """
    consent = st.checkbox(
        "Izinkan simpan data untuk retraining?",
        value=st.session_state.get('user_consent', False),
        help=(
            "Dengan mencentang ini, input teks dan hasil prediksi Anda akan "
            "disimpan ke database untuk meningkatkan model di masa depan. "
            "Data akan dianonimkan jika mengandung informasi pribadi."
        ),
        key='consent_checkbox'
    )
    
    # Update session state
    st.session_state['user_consent'] = consent
    
    return consent


def render_model_selector() -> str:
    """
    Display model selector untuk memilih antara model Indonesian (v1) dan IMDB English (v2).
    
    Returns:
        str: Model version ('v1' atau 'v2')
    """
    # Model options
    model_options = {
        'v1': '🇮🇩 Indonesian (3 kelas)',
        'v2': '🇺🇸 English IMDB (2 kelas)'
    }
    
    # Get current selection from session state
    current_version = st.session_state.get('selected_model_version', 'v1')
    
    # Model selector
    selected = st.selectbox(
        "Pilih Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(current_version),
        help="v1: Indonesian (negatif/netral/positif)\nv2: English IMDB (negative/positive)"
    )
    
    st.session_state['selected_model_version'] = selected
    
    # Display model info based on selection
    if selected == 'v1':
        st.info("🤖 **Model: Naive Bayes Indonesian**\n\n`MultinomialNB + TF-IDF`\n\nBahasa Indonesia - 3 kelas")
    else:
        st.info("🤖 **Model: Naive Bayes IMDB**\n\n`MultinomialNB + TF-IDF`\n\nEnglish - 2 kelas (binary)")
    
    # Display model metadata
    _display_model_metadata(selected)
    
    return selected


def _display_model_metadata(version: str):
    """
    Display model metadata.
    
    Args:
        version: Model version (e.g., 'v1')
    """
    metadata = MODEL_METADATA.get(version, {})
    
    if metadata:
        st.markdown("**Detail Model:**")
        
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
    st.markdown("---")
    st.markdown("### Retraining Pipeline")
    
    # Button untuk trigger retraining
    if st.button(
        "🔄 Latih Ulang Model",
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
            "Proses retraining akan:\n"
            "- Mengambil snapshot dataset dari database\n"
            "- Melatih model baru dengan data terbaru\n"
            "- Menyimpan model baru ke MLflow registry\n\n"
            "Proses ini mungkin memakan waktu beberapa menit.\n\n"
            "Apakah Anda yakin ingin melanjutkan?"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Ya, Lanjutkan", width="stretch"):
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


def render_sidebar(retraining_service=None):
    """
    Main function untuk render complete sidebar.
    
    Orchestrates semua sidebar components:
    - App title and description
    - Consent checkbox
    - Model selector
    - Retraining button
    
    Args:
        retraining_service: RetrainingService instance untuk handle retraining
    """
    with st.sidebar:
        # App title
        st.title(f"{settings.APP_ICON} {settings.APP_TITLE}")
        
        # 1. Panduan Singkat (Quick Guide)
        with st.expander("📚 Panduan Singkat", expanded=True):
            st.markdown("""
            **Selamat datang!**
            1. Pilih **Versi Model** yang ingin digunakan.
            2. Masukkan teks di area utama.
            3. Klik **Prediksi** untuk melihat hasil analisis sentimen.
            4. Cek tab **Monitoring** untuk melihat performa model.
            """)
        
        st.markdown("---")
        
        # 2. Pengaturan Model (Model Settings)
        st.subheader("🤖 Pengaturan Model")
        render_model_selector()
        
        st.markdown("---")
        
        # 3. Privasi & Data (Privacy & Data)
        st.subheader("🔒 Privasi & Data")
        render_consent_checkbox()
        
        # 4. Retraining (Advanced) - Dengan Autentikasi
        if retraining_service:
            st.markdown("---")
            st.subheader("🛠️ Maintenance (Advanced)")
            with st.expander("Menu Retraining", expanded=False):
                _render_admin_section(retraining_service)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<small>MLOps Streamlit Text AI v1.0</small>",
            unsafe_allow_html=True
        )


def _render_admin_section(retraining_service):
    """
    Render admin section dengan autentikasi password.
    
    Args:
        retraining_service: RetrainingService instance
    """
    # Check session timeout
    if _check_admin_session() and _check_session_timeout(timeout_minutes=30):
        _logout_admin()
        st.warning("⏰ Session timeout. Silakan login kembali.")
    
    # Check if admin is authenticated
    if not _check_admin_session():
        st.warning("🔐 **Area Terbatas**\n\nMenu ini memerlukan autentikasi admin.")
        
        # Login form
        with st.form("admin_login_form", clear_on_submit=True):
            st.markdown("**Login Admin**")
            password = st.text_input(
                "Password Admin", 
                type="password",
                placeholder="Masukkan password admin"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("🔑 Login", use_container_width=True)
            
            if submit:
                if password:
                    if _login_admin(password):
                        st.success("✅ Login berhasil!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Password salah!")
                        # Rate limiting - add delay for failed attempts
                        time.sleep(1)
                else:
                    st.error("❌ Masukkan password!")
    else:
        # Admin authenticated - show maintenance menu
        st.success("✅ **Authenticated as Admin**")
        
        # Show session info
        login_time = st.session_state.get('admin_login_time', 0)
        if login_time:
            elapsed = int((time.time() - login_time) / 60)
            remaining = 30 - elapsed
            st.caption(f"⏱️ Session tersisa: {remaining} menit")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            _logout_admin()
            st.info("Berhasil logout.")
            st.rerun()
        
        st.markdown("---")
        
        # Retraining section
        st.markdown("### 🔄 Retraining Pipeline")
        st.info(
            "**Perhatian:**\n"
            "- Proses ini akan melatih ulang model dengan data terbaru\n"
            "- Memerlukan data dengan user consent\n"
            "- Proses mungkin memakan waktu beberapa menit"
        )
        
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
