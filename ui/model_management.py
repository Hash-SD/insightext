"""
Model Management Page untuk Admin.

Halaman khusus untuk manajemen model AI dengan fitur:
1. Login Admin (di halaman ini)
2. Upload model baru (file .pkl)
3. Model Promotion (Staging → Production)
4. Archive Management
5. Model Comparison
6. Update History
7. Tutorial lengkap dalam Bahasa Indonesia
"""

import streamlit as st
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from config.settings import settings
from models.model_archiver import ModelArchiver
from models.model_updater import ModelUpdater


logger = logging.getLogger(__name__)


# ============================================================================
# ADMIN AUTHENTICATION FUNCTIONS
# ============================================================================

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


def render_admin_login_section():
    """Render section login admin di halaman Model Management."""
    
    # Check session timeout
    if _check_admin_session() and _check_session_timeout(timeout_minutes=30):
        _logout_admin()
        st.warning("⏰ Session timeout. Silakan login kembali.")
    
    # Admin status card
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if _check_admin_session():
            st.success("✅ **Admin Logged In** - Anda memiliki akses penuh ke fitur manajemen model")
        else:
            st.warning("🔒 **Login Required** - Masukkan password admin untuk mengakses fitur manajemen")
    
    with col2:
        if _check_admin_session():
            if st.button("🚪 Logout", use_container_width=True):
                _logout_admin()
                st.rerun()
        else:
            # Login form in expander
            with st.popover("🔐 Login Admin"):
                password = st.text_input(
                    "Password Admin",
                    type="password",
                    placeholder="Masukkan password",
                    key="mgmt_admin_pass"
                )
                
                if st.button("🔓 Masuk", use_container_width=True):
                    if password:
                        if _login_admin(password):
                            st.success("✅ Login berhasil!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Password salah!")
                    else:
                        st.error("⚠️ Masukkan password!")
    
    return _check_admin_session()


# ============================================================================
# TUTORIAL SECTION
# ============================================================================

def render_tutorial_section():
    """Render tutorial section untuk admin."""
    
    with st.expander("📚 **TUTORIAL: Cara Upload & Simpan Model** (Klik untuk membuka)", expanded=False):
        st.markdown("""
        ## 📖 Panduan Lengkap Manajemen Model untuk Admin
        
        ---
        
        ### 🎯 **LANGKAH 1: Persiapan File Model**
        
        Sebelum upload, pastikan Anda memiliki file-file berikut:
        
        | File | Keterangan | Wajib? |
        |------|------------|--------|
        | `model_pipeline.pkl` | File model utama (Naive Bayes + TF-IDF) | ✅ Ya |
        | `preprocessor.pkl` | File preprocessor untuk text cleaning | ⚪ Opsional |
        | `training_config.json` | Konfigurasi dan metrics model | ⚪ Auto-generate |
        
        **Format yang didukung:** `.pkl` (pickle file)
        
        ---
        
        ### 🔐 **LANGKAH 2: Login sebagai Admin**
        
        1. Klik tombol **"🔐 Login Admin"** di pojok kanan atas halaman ini
        2. Masukkan **password admin**
        3. Klik tombol **"🔓 Masuk"**
        4. Jika berhasil, status akan berubah menjadi **"✅ Admin Logged In"**
        
        ⚠️ **Penting:** Tanpa login admin, Anda tidak bisa upload atau restore model!
        
        ---
        
        ### 📤 **LANGKAH 3: Upload Model Baru**
        
        1. Pilih tab **"📤 Upload Model"** di bawah
        2. Klik **"Browse files"** pada bagian **"Upload File Model"**
        3. Pilih file `.pkl` model Anda
        4. (Opsional) Upload juga file preprocessor
        5. Isi **Metrics Model**:
           - **Akurasi Model**: Nilai 0-1 (contoh: 0.75 = 75%)
           - **F1 Score**: Nilai 0-1 (contoh: 0.73 = 73%)
           - **Training Samples**: Jumlah data training (contoh: 1000)
        6. Tulis **Alasan Update** (contoh: "Model baru dengan balanced data")
        7. Klik tombol **"🚀 Update Model Sekarang"**
        
        ---
        
        ### ✅ **LANGKAH 4: Verifikasi Model**
        
        Setelah upload berhasil:
        1. Buka halaman **"🔮 Prediksi"**
        2. Pilih versi model yang baru diupload
        3. Coba lakukan prediksi untuk memastikan model berfungsi
        
        ---
        
        ### 🔄 **FITUR TAMBAHAN**
        
        | Fitur | Keterangan |
        |-------|------------|
        | **🚀 Model Promotion** | Promosikan model dari staging ke production |
        | **📦 Archive Management** | Lihat & kelola model lama yang di-backup |
        | **🔄 Restore** | Kembalikan model lama jika ada masalah |
        | **⚖️ Model Comparison** | Bandingkan performa model lama vs baru |
        | **📋 Update History** | Lihat riwayat semua update model |
        
        ---
        
        ### ⚠️ **CATATAN PENTING**
        
        - ✅ Model lama akan **otomatis di-backup** sebelum diganti
        - ✅ Anda bisa **rollback/restore** kapan saja jika ada masalah
        - ✅ Semua update tercatat di **Update History**
        - ❌ Jangan upload file yang bukan format `.pkl`
        - ❌ Pastikan metrics yang diisi akurat untuk tracking
        
        ---
        
        ### 🆘 **Butuh Bantuan?**
        
        Jika mengalami error saat upload:
        1. Pastikan file `.pkl` valid dan tidak corrupt
        2. Pastikan sudah login sebagai admin
        3. Cek koneksi internet (jika menggunakan cloud database)
        4. Hubungi tim teknis jika masalah berlanjut
        """)


# ============================================================================
# UPLOAD MODEL SECTION
# ============================================================================

def render_upload_model_tab(is_admin: bool, updater: ModelUpdater, archiver: ModelArchiver):
    """Render tab untuk upload model baru."""
    
    st.markdown("#### 📤 Upload Model Baru")
    
    if not is_admin:
        st.info("🔒 Login sebagai admin untuk mengupload model baru")
        return
    
    st.markdown("""
    Upload model baru untuk menggantikan model production saat ini.
    Model lama akan otomatis di-archive sebelum diganti.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Upload File Model Baru:**")
        uploaded_model = st.file_uploader(
            "Pilih file model (.pkl)",
            type=['pkl'],
            help="Upload model pipeline yang sudah di-train",
            key="upload_model_file"
        )
        
        if uploaded_model is not None:
            st.success(f"✓ File diterima: {uploaded_model.name}")
    
    with col2:
        st.markdown("**Upload File Preprocessor:**")
        uploaded_preprocessor = st.file_uploader(
            "Pilih file preprocessor (.pkl)",
            type=['pkl'],
            help="Upload preprocessor/vectorizer (opsional)",
            key="upload_preprocessor_file"
        )
        
        if uploaded_preprocessor is not None:
            st.success(f"✓ File diterima: {uploaded_preprocessor.name}")
    
    # Model metrics input
    st.markdown("---")
    st.markdown("**📈 Metrics Model Baru:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_accuracy = st.number_input(
            "Akurasi Model",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.01,
            help="Akurasi model baru pada test set (0-1)",
            key="new_accuracy"
        )
    
    with col2:
        new_f1_score = st.number_input(
            "F1 Score",
            min_value=0.0,
            max_value=1.0,
            value=0.73,
            step=0.01,
            help="F1 Score model baru (0-1)",
            key="new_f1"
        )
    
    with col3:
        new_training_samples = st.number_input(
            "Training Samples",
            min_value=100,
            value=1000,
            step=100,
            help="Jumlah samples yang digunakan untuk training",
            key="new_samples"
        )
    
    # Update reason
    update_reason = st.text_area(
        "📝 Alasan Update Model:",
        value="",
        placeholder="Contoh: Model baru dengan balanced data menggunakan oversampling technique",
        help="Catatan tentang mengapa model di-update",
        key="update_reason"
    )
    
    st.markdown("---")
    
    # Update button
    if st.button("🚀 Update Model Sekarang", use_container_width=True, type="primary", key="btn_update"):
        if uploaded_model is not None:
            with st.spinner("⏳ Memproses update model..."):
                try:
                    # Create temporary directory untuk uploaded files
                    temp_model_dir = Path('temp_model_upload')
                    temp_model_dir.mkdir(exist_ok=True)
                    
                    # Save uploaded files
                    model_path = temp_model_dir / uploaded_model.name
                    with open(model_path, 'wb') as f:
                        f.write(uploaded_model.getvalue())
                    
                    if uploaded_preprocessor is not None:
                        preprocessor_path = temp_model_dir / uploaded_preprocessor.name
                        with open(preprocessor_path, 'wb') as f:
                            f.write(uploaded_preprocessor.getvalue())
                    
                    # Prepare metrics
                    new_metrics = {
                        'accuracy': new_accuracy,
                        'f1_score': new_f1_score,
                        'training_samples': new_training_samples,
                        'uploaded_at': datetime.now().isoformat()
                    }
                    
                    # Update model
                    success, report = updater.update_model_v1(
                        new_model_path=str(temp_model_dir),
                        new_metrics=new_metrics,
                        update_reason=update_reason or "Model update via UI",
                        auto_validate=True
                    )
                    
                    if success:
                        st.success("✅ Model berhasil di-update!")
                        st.balloons()
                        st.json(report.get('summary', {}))
                    else:
                        st.error(f"❌ Update gagal: {report.get('error', 'Unknown error')}")
                        with st.expander("Detail Error"):
                            st.json(report)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Model upload error: {e}", exc_info=True)
        else:
            st.warning("⚠️ Silakan upload file model terlebih dahulu")


# ============================================================================
# MODEL PROMOTION SECTION
# ============================================================================

def render_promotion_tab(is_admin: bool, updater: ModelUpdater, archiver: ModelArchiver, current_version: str):
    """Render tab untuk model promotion."""
    
    st.markdown("#### 🚀 Model Promotion")
    st.markdown("Promosikan model antar stage untuk deployment yang terkelola.")
    
    # Current model status
    st.markdown("##### 🎯 Status Model Saat Ini")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🇮🇩 Model v1 (Indonesian)**
        - Stage: `Production`
        - Status: ✅ Aktif
        """)
    
    with col2:
        st.info("""
        **🇺🇸 Model v2 (English)**
        - Stage: `Production`
        - Status: ✅ Aktif
        """)
    
    with col3:
        staging_models = archiver.list_archived_models()
        st.info(f"""
        **📦 Archived Models**
        - Total: {len(staging_models)}
        - Siap restore: {len(staging_models)}
        """)
    
    if not is_admin:
        st.info("🔒 Login sebagai admin untuk melakukan promosi model")
        return
    
    st.markdown("---")
    st.markdown("##### 🔄 Aksi Promosi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Archive → Production**")
        st.caption("Restore model dari archive ke production")
        
        staging_models_list = archiver.list_archived_models()
        
        if staging_models_list:
            selected_staging = st.selectbox(
                "Pilih model dari archive:",
                options=range(len(staging_models_list)),
                format_func=lambda i: f"{staging_models_list[i]['version']} - {staging_models_list[i]['archived_at'][:10]}",
                key="promo_staging_select"
            )
            
            if st.button("⬆️ Restore ke Production", use_container_width=True, key="btn_promote"):
                with st.spinner("Mempromosikan model..."):
                    try:
                        selected_model = staging_models_list[selected_staging]
                        success, result = updater.rollback_to_archive(selected_model['path'])
                        
                        if success:
                            st.success("✅ Model berhasil di-restore ke Production!")
                            st.json(result)
                        else:
                            st.error("❌ Restore gagal")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("📭 Tidak ada model di archive")
    
    with col2:
        st.markdown("**Production → Archive**")
        st.caption("Backup model production saat ini ke archive")
        
        archive_notes = st.text_input(
            "Catatan archive:",
            placeholder="Contoh: Backup sebelum update",
            key="promo_archive_notes"
        )
        
        if st.button("⬇️ Archive Model Production", use_container_width=True, key="btn_archive"):
            with st.spinner("Mengarchive model..."):
                try:
                    current_config_path = Path('models/saved_model/training_config.json')
                    current_metrics = {}
                    if current_config_path.exists():
                        with open(current_config_path, 'r') as f:
                            config = json.load(f)
                        current_metrics = config.get('metrics', {})
                    
                    archive_path = archiver.archive_model(
                        version=current_version,
                        current_model_path='models/saved_model',
                        metrics=current_metrics,
                        notes=archive_notes or "Manual archive from production"
                    )
                    
                    st.success(f"✅ Model berhasil di-archive!")
                    st.info(f"📁 Lokasi: `{archive_path}`")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# ============================================================================
# ARCHIVE MANAGEMENT SECTION
# ============================================================================

def render_archive_tab(is_admin: bool, updater: ModelUpdater, archiver: ModelArchiver):
    """Render tab untuk archive management."""
    
    st.markdown("#### 📦 Archive Management")
    st.markdown("Kelola versi model lama yang sudah di-archive.")
    
    archived_models = archiver.list_archived_models()
    
    if not archived_models:
        st.info("📭 Belum ada model yang di-archive")
        return
    
    st.markdown(f"**Total Archive: {len(archived_models)} versi**")
    
    for idx, archive_info in enumerate(archived_models):
        with st.expander(
            f"📦 {archive_info['version']} - {archive_info['archived_at'][:10]}",
            expanded=(idx == 0)
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Metadata:**")
                st.text(f"Timestamp: {archive_info['archived_at']}")
                st.text(f"Notes: {archive_info.get('notes', 'N/A')}")
            
            with col2:
                st.markdown("**Metrics:**")
                metrics = archive_info.get('metrics', {})
                if metrics:
                    st.text(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
                    st.text(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
                else:
                    st.info("Tidak ada metrics tersimpan")
            
            if is_admin:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Restore", key=f"arch_restore_{idx}"):
                        success, result = updater.rollback_to_archive(archive_info['path'])
                        if success:
                            st.success("✅ Model restored!")
                            st.rerun()
                        else:
                            st.error("❌ Restore failed")
                
                with col2:
                    if st.button("👁️ View Files", key=f"arch_view_{idx}"):
                        archive_detail = archiver.get_archive_info(archive_info['path'])
                        if archive_detail:
                            st.text(f"Files: {', '.join(archive_detail.get('files', []))}")
                
                with col3:
                    if st.button("🗑️ Delete", key=f"arch_delete_{idx}"):
                        success = archiver.delete_archive(archive_info['path'])
                        if success:
                            st.success("✅ Archive deleted")
                            st.rerun()
                        else:
                            st.error("❌ Delete failed")
            else:
                st.info("🔒 Login admin untuk restore/delete")


# ============================================================================
# MODEL COMPARISON SECTION
# ============================================================================

def render_comparison_tab(archiver: ModelArchiver):
    """Render tab untuk model comparison."""
    
    st.markdown("#### ⚖️ Model Comparison")
    st.markdown("Bandingkan performa model production dengan model di archive.")
    
    # Get current model metrics
    try:
        current_config_path = Path('models/saved_model/training_config.json')
        if current_config_path.exists():
            with open(current_config_path, 'r') as f:
                current_config = json.load(f)
            current_metrics = current_config.get('metrics', {})
        else:
            current_metrics = {'accuracy': 0.6972, 'f1_score': 0.6782}  # Default
    except:
        current_metrics = {'accuracy': 0.6972, 'f1_score': 0.6782}
    
    st.markdown("**Model Production Saat Ini:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Akurasi", f"{current_metrics.get('accuracy', 0):.2%}")
    with col2:
        st.metric("F1 Score", f"{current_metrics.get('f1_score', 0):.2%}")
    with col3:
        samples = current_metrics.get('training_samples', 'N/A')
        st.metric("Training Samples", f"{samples:,}" if isinstance(samples, int) else samples)
    
    st.markdown("---")
    
    archived_models = archiver.list_archived_models()
    
    if not archived_models:
        st.info("Belum ada model archive untuk dibandingkan")
        return
    
    selected_archive = st.selectbox(
        "Pilih model archive untuk dibandingkan:",
        options=range(len(archived_models)),
        format_func=lambda i: f"{archived_models[i]['version']} - {archived_models[i]['archived_at'][:10]}",
        key="compare_select"
    )
    
    archive_metrics = archived_models[selected_archive].get('metrics', {})
    
    st.markdown("**Model Archive (Dipilih):**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Akurasi", f"{archive_metrics.get('accuracy', 0):.2%}")
    with col2:
        st.metric("F1 Score", f"{archive_metrics.get('f1_score', 0):.2%}")
    with col3:
        samples = archive_metrics.get('training_samples', 'N/A')
        st.metric("Training Samples", f"{samples:,}" if isinstance(samples, int) else samples)
    
    # Comparison summary
    st.markdown("---")
    st.markdown("**📋 Ringkasan Perbandingan:**")
    
    acc_diff = current_metrics.get('accuracy', 0) - archive_metrics.get('accuracy', 0)
    f1_diff = current_metrics.get('f1_score', 0) - archive_metrics.get('f1_score', 0)
    
    col1, col2 = st.columns(2)
    with col1:
        delta_color = "normal" if acc_diff >= 0 else "inverse"
        st.metric("Δ Akurasi", f"{acc_diff:+.2%}", delta="Better" if acc_diff > 0 else "Worse" if acc_diff < 0 else "Same")
    with col2:
        st.metric("Δ F1 Score", f"{f1_diff:+.2%}", delta="Better" if f1_diff > 0 else "Worse" if f1_diff < 0 else "Same")


# ============================================================================
# UPDATE HISTORY SECTION
# ============================================================================

def render_history_tab(updater: ModelUpdater):
    """Render tab untuk update history."""
    
    st.markdown("#### 📋 Update History")
    st.markdown("Riwayat semua update model yang pernah dilakukan.")
    
    history = updater.list_update_history(limit=20)
    
    if not history:
        st.info("📭 Belum ada history update")
        return
    
    st.markdown(f"**Total Updates: {len(history)}**")
    
    for idx, record in enumerate(history):
        status_icon = "✅" if record.get('success') else "❌"
        timestamp = record.get('timestamp', 'N/A')
        if timestamp != 'N/A':
            timestamp = timestamp[:19]  # Trim to datetime only
        
        with st.expander(f"{status_icon} {timestamp} - {record.get('reason', 'N/A')[:40]}..."):
            st.text(f"File: {record.get('file', 'N/A')}")
            st.text(f"Status: {'Success' if record.get('success') else 'Failed'}")
            st.text(f"Reason: {record.get('reason', 'N/A')}")


# ============================================================================
# MAIN PAGE RENDER
# ============================================================================

def render_model_management_page():
    """Main function untuk render halaman Model Management."""
    
    st.title("🚀 Model Management")
    st.caption("Kelola model AI untuk sistem prediksi sentimen")
    
    st.markdown("---")
    
    # Admin Login Section
    is_admin = render_admin_login_section()
    
    st.markdown("---")
    
    # Tutorial Section
    render_tutorial_section()
    
    st.markdown("---")
    
    # Initialize services
    updater = ModelUpdater()
    archiver = ModelArchiver()
    current_version = st.session_state.get('selected_model_version', 'v1')
    
    # Tabs for different features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload Model",
        "🚀 Model Promotion",
        "📦 Archive",
        "⚖️ Comparison",
        "📋 History"
    ])
    
    with tab1:
        render_upload_model_tab(is_admin, updater, archiver)
    
    with tab2:
        render_promotion_tab(is_admin, updater, archiver, current_version)
    
    with tab3:
        render_archive_tab(is_admin, updater, archiver)
    
    with tab4:
        render_comparison_tab(archiver)
    
    with tab5:
        render_history_tab(updater)
