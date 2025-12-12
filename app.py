"""
Main Streamlit application untuk Sistem AI Berbasis Teks.
Entry point untuk aplikasi MLOps dengan redesign modern (Material Design).
"""

import streamlit as st
import logging
from config.settings import settings
from utils.logger import setup_logger, log_error

# Import database & services
from database.db_manager import DatabaseManager
from database.db_manager_supabase import SupabaseDatabaseManager
from models.model_loader import ModelLoader
from services.prediction_service import PredictionService
from services.monitoring_service import MonitoringService
from services.retraining_service import RetrainingService

# Import New UI Components
from ui.sidebar import render_sidebar
from ui.main_area import render_input_section, render_results_section, render_prediction_button, render_prediction_history
from ui.monitoring import render_monitoring_dashboard
from ui.model_management import render_model_management_page
from ui.styles import load_css

# Setup logger
logger = setup_logger(
    name='mlops_app',
    log_file=settings.LOG_FILE,
    level=getattr(logging, settings.LOG_LEVEL.upper())
)

def initialize_session_state():
    """Initialize session state defaults."""
    defaults = {
        'selected_model_version': settings.DEFAULT_MODEL_VERSION,
        'user_consent': True,
        'dont_save_data': False,
        'prediction_history': [],
        'current_prediction': None,
        'retraining_status': 'idle',
        'text_input_area': "", # Changed from text_input to match main_area.py key
        'user_mode': 'Beginner',
        'show_retrain_confirmation': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def initialize_resources():
    """
    Initialize DB and Model Loader. Cached resource.
    Returns: (db_manager, model_loader) tuple
    """
    # 1. Database
    try:
        if settings.is_supabase():
            if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
                raise ValueError("Missing Supabase credentials")
            db_manager = SupabaseDatabaseManager(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            if not db_manager.connect(): raise Exception("Failed to connect to Supabase")
        else:
            db_manager = DatabaseManager(settings.get_database_path())
            db_manager.connect()
            db_manager.initialize_schema()
    except Exception as e:
        logger.error(f"DB Init failed: {e}")
        # Fallback
        db_manager = DatabaseManager("mlops_app.db")
        db_manager.connect()
        db_manager.initialize_schema()
        
    # 2. Model Loader
    try:
        model_loader = ModelLoader(mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI)
    except:
        model_loader = ModelLoader(mlflow_tracking_uri=None)
        
    return db_manager, model_loader

def main():
    # 1. Page Config
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon=settings.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 2. Load Custom CSS
    load_css()
    
    # 3. Initialization
    initialize_session_state()
    
    with st.spinner("🚀 Memuat sistem..."):
        db_manager, model_loader = initialize_resources()
        
    # Initialize Services
    prediction_service = PredictionService(db_manager, model_loader)
    monitoring_service = MonitoringService(db_manager)
    retraining_service = RetrainingService(db_manager, settings.MLFLOW_TRACKING_URI)
    
    # 4. Render Sidebar
    selected_page = render_sidebar(retraining_service)
    
    # 5. Page Routing
    if selected_page == "🔮 Prediksi":
        st.markdown(f"## {settings.APP_TITLE}")
        
        # --- NEW LAYOUT: 2 Columns ---
        col_input, col_result = st.columns([1, 1], gap="large")
        
        with col_input:
            # 1. Render Input Card
            text_input = render_input_section() # Returns text
            
            # 2. Render Action Button
            is_valid = len(text_input) >= settings.MIN_INPUT_LENGTH
            if not is_valid and text_input:
                st.warning(f"Minimal {settings.MIN_INPUT_LENGTH} karakter.")
                
            button_clicked = render_prediction_button(enabled=is_valid)
            
            # 3. Validation & Logic
            if button_clicked and is_valid:
                with st.spinner("🤖 Menganalisis sentimen..."):
                    try:
                        result = prediction_service.predict(
                            text=text_input,
                            model_version=st.session_state.selected_model_version,
                            user_consent=st.session_state.user_consent
                        )
                        st.session_state.current_prediction = result
                        
                        # Add to local history display immediately
                        if 'prediction_history' not in st.session_state: st.session_state.prediction_history = []
                        st.session_state.prediction_history.insert(0, result)
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.error(f"Prediction error: {e}")
        
        with col_result:
            # 4. Render Result Card
            # Always render the result section, it handles empty state internally
            render_results_section(st.session_state.current_prediction)
            
        # 5. History Section (Full Width below)
        st.markdown("---")
        st.subheader("📜 Riwayat")
        # Reuse existing logic for history (simplified)
        try:
             history = db_manager.get_recent_predictions(limit=5)
             if history:
                 # Simplified table or reuse component
                 # We can reuse the old component or just show simple table
                 st.dataframe(
                     [{'Waktu': h['timestamp'], 'Text': h['text_input'], 'Prediksi': h['prediction']} for h in history],
                     use_container_width=True,
                     hide_index=True
                 )
             else:
                 st.info("Belum ada riwayat.")
        except:
             pass

    elif selected_page == "📊 Monitoring":
        st.title("📊 Monitoring Dashboard")
        st.markdown("---")
        render_monitoring_dashboard(monitoring_service)
        
    elif selected_page == "🚀 Model Management":
        st.title("🚀 Model Management") 
        # Note: We need to import the admin logic or keep it here.
        # Ideally, we should have refactored the admin logic into the page function itself.
        # But previous code had logic in sidebar? No, render_model_management_page handles it.
        render_model_management_page()

if __name__ == "__main__":
    main()
