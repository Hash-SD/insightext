"""
Monitoring dashboard components untuk MLOps Streamlit Text AI application.

Module ini menyediakan fungsi untuk render monitoring dashboard dengan:
- Metrics table per model version
- Latency histogram
- Drift score display
- Model promotion buttons (placeholder)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any, Optional
from config.settings import settings


# Model metadata for Naive Bayes
MODEL_METADATA = {
    'v1': {
        'name': 'Naive Bayes Sentiment Analysis',
        'model_type': 'MultinomialNB + TF-IDF',
        'task': 'Sentiment Analysis',
        'labels': ['negatif', 'netral', 'positif'],
        'accuracy': 0.6972,
        'f1_score': 0.6782,
        'description': 'Model sentiment analysis berbasis Naive Bayes dengan TF-IDF untuk teks Bahasa Indonesia'
    }
}


def render_metrics_table(metrics_summary: Dict[str, Dict[str, Any]]):
    """
    Render metrics table untuk display accuracy per model version.
    
    Args:
        metrics_summary: Dictionary dengan structure:
            {
                'v1': {
                    'prediction_count': int,
                    'avg_confidence': float,
                    'avg_latency': float,
                    ...
                },
                ...
            }
    """
    st.markdown("### 📊 Metrik Model")
    
    if not metrics_summary:
        st.info("Belum ada data metrik tersedia")
        return
    
    # Prepare data untuk table
    table_data = []
    
    for version in settings.MODEL_VERSIONS:
        # Get metadata dari placeholders
        metadata = MODEL_METADATA.get(version, {})
        
        # Get metrics dari summary (jika ada)
        metrics = metrics_summary.get(version, {})
        
        table_data.append({
            'Versi': version,
            'Nama Model': metadata.get('name', 'N/A'),
            'Akurasi': f"{metadata.get('accuracy', 0.0):.1%}",
            'F1 Score': f"{metadata.get('f1_score', 0.0):.1%}",
            'Jumlah Prediksi': metrics.get('prediction_count', 0),
            'Avg Confidence': f"{metrics.get('avg_confidence', 0.0):.1%}" if metrics.get('avg_confidence') else 'N/A',
            'Avg Latency (ms)': f"{metrics.get('avg_latency', 0.0)*1000:.2f}" if metrics.get('avg_latency') else 'N/A'
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Display table
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            'Versi': st.column_config.TextColumn('Versi', width='small'),
            'Nama Model': st.column_config.TextColumn('Nama Model', width='large'),
            'Akurasi': st.column_config.TextColumn('Akurasi', width='small'),
            'F1 Score': st.column_config.TextColumn('F1 Score', width='small'),
            'Jumlah Prediksi': st.column_config.NumberColumn('Jumlah Prediksi', width='small'),
            'Avg Confidence': st.column_config.TextColumn('Avg Confidence', width='small'),
            'Avg Latency (ms)': st.column_config.TextColumn('Avg Latency (ms)', width='small')
        }
    )


def render_latency_histogram(latency_data: List[float], model_version: Optional[str] = None):
    """
    Render latency histogram menggunakan Plotly.
    
    Args:
        latency_data: List of latency values dalam seconds
        model_version: Specific model version untuk filter, atau None untuk all
    """
    st.markdown("### ⏱️ Distribusi Latency")
    
    if not latency_data or len(latency_data) == 0:
        st.info("Belum ada data latency tersedia")
        return
    
    # Convert to milliseconds
    latency_ms = [lat * 1000 for lat in latency_data]
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=latency_ms,
        nbinsx=20,
        marker=dict(
            color='#007bff',
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>Latency:</b> %{x:.2f} ms<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    # Add threshold line
    threshold_ms = settings.LATENCY_THRESHOLD_MS
    fig.add_vline(
        x=threshold_ms,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold_ms:.0f} ms",
        annotation_position="top right"
    )
    
    fig.update_layout(
        xaxis_title="Latency (ms)",
        yaxis_title="Jumlah Prediksi",
        title=f"Distribusi Latency {f'untuk {model_version}' if model_version else '(Semua Model)'}",
        showlegend=False,
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Min", f"{min(latency_ms):.2f} ms")
    
    with col2:
        st.metric("Rata-rata", f"{sum(latency_ms)/len(latency_ms):.2f} ms")
    
    with col3:
        st.metric("Max", f"{max(latency_ms):.2f} ms")
    
    with col4:
        # Count predictions above threshold
        above_threshold = sum(1 for lat in latency_ms if lat > threshold_ms)
        st.metric("Di Atas Threshold", f"{above_threshold}")


def render_drift_score(drift_score: float):
    """
    Render drift score dengan st.metric.
    
    Args:
        drift_score: Drift score value (0-1, higher = more drift)
    """
    st.markdown("### 📈 Data Drift Detection")
    
    # Determine status based on drift score
    if drift_score < 0.2:
        status = "Rendah"
        color = "green"
        delta_color = "normal"
    elif drift_score < 0.4:
        status = "Sedang"
        color = "orange"
        delta_color = "off"
    else:
        status = "Tinggi"
        color = "red"
        delta_color = "inverse"
    
    st.metric(
        label="Drift Score",
        value=f"{drift_score:.2%}",
        delta=status,
        delta_color=delta_color,
        help="Skor drift menunjukkan seberapa banyak distribusi data saat ini berbeda dari data training"
    )
    
    st.markdown(f"**Status:** <span style='color: {color};'>{status}</span>", unsafe_allow_html=True)
    st.markdown(
        "Drift score mengukur perubahan distribusi data. "
        "Skor tinggi menandakan model mungkin perlu di-retrain."
    )
    
    # Progress bar untuk visual representation
    st.progress(min(drift_score, 1.0))


def render_promotion_buttons(current_version: str):
    """
    Render buttons untuk model stage promotion (placeholder).
    
    Args:
        current_version: Current selected model version
    """
    st.markdown("### 🚀 Model Promotion")
    
    st.info(
        "**Placeholder Feature**\n\n"
        "Fitur ini akan memungkinkan promosi model antara stages:\n"
        "- Staging → Production\n"
        "- Production → Archived\n\n"
        "Integrasi dengan MLflow Model Registry akan diimplementasikan di fase berikutnya."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(
            "📤 Promote to Staging",
            width="stretch",
            disabled=True,
            help="Promosikan model ke Staging (Coming Soon)"
        ):
            st.warning("Fitur ini akan segera tersedia")
    
    with col2:
        if st.button(
            "✅ Promote to Production",
            width="stretch",
            disabled=True,
            help="Promosikan model ke Production (Coming Soon)"
        ):
            st.warning("Fitur ini akan segera tersedia")
    
    with col3:
        if st.button(
            "📦 Archive Model",
            width="stretch",
            disabled=True,
            help="Arsipkan model (Coming Soon)"
        ):
            st.warning("Fitur ini akan segera tersedia")


def render_prediction_distribution(metrics_summary: Dict[str, Dict[str, Any]]):
    """
    Render prediction distribution chart per model version.
    
    Args:
        metrics_summary: Metrics summary dari monitoring service
    """
    st.markdown("### 📊 Distribusi Prediksi per Model")
    
    if not metrics_summary:
        st.info("Belum ada data distribusi tersedia")
        return
    
    # Prepare data untuk chart
    versions = []
    counts = []
    
    for version in settings.MODEL_VERSIONS:
        metrics = metrics_summary.get(version, {})
        count = metrics.get('prediction_count', 0)
        
        if count > 0:
            versions.append(version)
            counts.append(count)
    
    if not versions:
        st.info("Belum ada prediksi yang dilakukan")
        return
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=versions,
        y=counts,
        marker=dict(
            color='#007bff',
            line=dict(color='white', width=1)
        ),
        text=counts,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Prediksi: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Versi Model",
        yaxis_title="Jumlah Prediksi",
        title="Jumlah Prediksi per Versi Model",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")


def render_monitoring_dashboard(monitoring_service):
    """
    Main function untuk render complete monitoring dashboard.
    
    Args:
        monitoring_service: MonitoringService instance untuk fetch data
    """
    st.markdown("## 📊 Monitoring Dashboard")
    st.markdown("Monitor performa model dan deteksi anomali secara real-time.")
    
    try:
        # Fetch metrics summary
        with st.spinner("⏳ Memuat data monitoring..."):
            metrics_summary = monitoring_service.get_metrics_summary()
            latency_data = monitoring_service.get_latency_distribution()
            drift_score = monitoring_service.calculate_drift_score()
            selected_version = st.session_state.get('selected_model_version')
        
        # Top Level Summary Metrics
        total_predictions = sum(m.get('prediction_count', 0) for m in metrics_summary.values())
        avg_latency_all = 0
        if latency_data:
            avg_latency_all = sum(latency_data) / len(latency_data) * 1000
            
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Prediksi", total_predictions)
        col2.metric("Rata-rata Latency", f"{avg_latency_all:.2f} ms")
        col3.metric("Drift Score Global", f"{drift_score:.1%}", delta_color="inverse")
        
        st.markdown("---")

        # Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📈 Overview", "⏱️ Performa & Latency", "🚀 Model Management"])
        
        with tab1:
            st.subheader("Ringkasan Metrik Model")
            render_metrics_table(metrics_summary)
            
            st.markdown("---")
            render_prediction_distribution(metrics_summary)
            
            st.markdown("---")
            render_drift_score(drift_score)

        with tab2:
            st.subheader("Analisis Latency")
            render_latency_histogram(latency_data, selected_version)
            
        with tab3:
            st.subheader("Manajemen Model")
            if selected_version:
                render_promotion_buttons(selected_version)
            else:
                st.info("Pilih versi model di sidebar untuk melihat opsi manajemen.")
            
    except ConnectionError as e:
        st.error(f"❌ Gagal terhubung ke database: {str(e)}")
        st.info("💡 Periksa koneksi database dan coba refresh halaman")
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Database connection error in monitoring: {e}", exc_info=True)
    except Exception as e:
        st.error(f"❌ Gagal memuat dashboard monitoring: {str(e)}")
        st.info("💡 Silakan refresh halaman atau hubungi administrator")
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error rendering monitoring dashboard: {e}", exc_info=True)
