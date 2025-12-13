"""Monitoring dashboard components for MLOps Streamlit Text AI application."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Model metadata
MODEL_METADATA = {
    'v1': {
        'name': 'NB Indonesian Sentiment',
        'model_type': 'MultinomialNB + TF-IDF',
        'task': 'Sentiment Analysis',
        'labels': ['negatif', 'netral', 'positif'],
        'accuracy': 0.6972,
        'f1_score': 0.6782,
        'description': 'Analisis sentimen Bahasa Indonesia (3 kelas)'
    },
    'v2': {
        'name': 'NB English Sentiment',
        'model_type': 'MultinomialNB + TF-IDF',
        'task': 'Sentiment Analysis',
        'labels': ['negative', 'positive'],
        'accuracy': 0.8647,
        'f1_score': 0.8647,
        'description': 'Analisis sentimen English (binary)'
    }
}


@st.cache_data(ttl=300)
def _get_training_config(config_path: str) -> Dict[str, Any]:
    """Read and cache training configuration file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def render_metrics_table(metrics_summary: Dict[str, Dict[str, Any]]):
    """Render metrics table for model version accuracy."""
    if not metrics_summary:
        st.info("Belum ada data metrik tersedia")
        return
    
    html = """
    <div class="glass-card">
        <h3 style="margin-top: 0; margin-bottom: 20px;">Evaluasi Model</h3>
        <table class="glass-table">
            <thead>
                <tr>
                    <th style="width: 10%;">Versi</th>
                    <th style="width: 25%;">Nama Model</th>
                    <th style="width: 15%;">Akurasi</th>
                    <th style="width: 15%;">F1 Score</th>
                    <th style="width: 15%;">Prediksi</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for version in settings.MODEL_VERSIONS:
        metadata = MODEL_METADATA.get(version, {})
        metrics = metrics_summary.get(version, {})
        
        acc = f"{metadata.get('accuracy', 0.0):.1%}"
        f1 = f"{metadata.get('f1_score', 0.0):.1%}"
        count = metrics.get('prediction_count', 0)
        name = metadata.get('name', 'N/A')
        
        html += f"""<tr>
            <td><span class="badge-neu" style="font-weight: bold;">{version}</span></td>
            <td>{name}</td>
            <td style="color: #166534; font-weight: 500;">{acc}</td>
            <td style="color: #15803d; font-weight: 500;">{f1}</td>
            <td>{count:,}</td>
        </tr>"""
    
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_latency_histogram(latency_data: List[float], model_version: Optional[str] = None):
    """Render latency histogram using Plotly."""
    st.markdown("### ⏱️ Distribusi Latency")
    
    if not latency_data:
        st.info("Belum ada data latency tersedia")
        return
    
    latency_ms = [lat * 1000 for lat in latency_data]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=latency_ms,
        nbinsx=20,
        marker=dict(color='#1a73e8', line=dict(color='white', width=1)),
        hovertemplate='<b>Latency:</b> %{x:.2f} ms<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    threshold_ms = settings.LATENCY_THRESHOLD_MS
    fig.add_vline(
        x=threshold_ms,
        line_dash="dash",
        line_color="#d93025",
        annotation_text=f"Threshold: {threshold_ms:.0f} ms",
        annotation_position="top right"
    )
    
    title = f"Distribusi Latency {f'untuk {model_version}' if model_version else '(Semua Model)'}"
    fig.update_layout(
        xaxis_title="Latency (ms)",
        yaxis_title="Jumlah Prediksi",
        title=title,
        showlegend=False,
        height=350,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min", f"{min(latency_ms):.2f} ms")
    col2.metric("Rata-rata", f"{sum(latency_ms)/len(latency_ms):.2f} ms")
    col3.metric("Max", f"{max(latency_ms):.2f} ms")
    
    above_threshold = sum(1 for lat in latency_ms if lat > threshold_ms)
    col4.metric("Di Atas Threshold", f"{above_threshold}", delta_color="inverse")


def render_drift_score(drift_score: float):
    """Render drift score with pure HTML."""
    if drift_score < 0.2:
        status, color, bg_color, bar_color = "Rendah", "#166534", "#DCFCE7", "#22c55e"
    elif drift_score < 0.4:
        status, color, bg_color, bar_color = "Sedang", "#854d0e", "#FEF9C3", "#eab308"
    else:
        status, color, bg_color, bar_color = "Tinggi", "#991B1B", "#FEE2E2", "#ef4444"
    
    progress_width = min(drift_score * 100, 100)
    
    html = f"""
<div class="glass-card">
<h3 style="margin-top: 0; margin-bottom: 20px;">📈 Data Drift Detection</h3>
<div style="display: flex; gap: 40px; align-items: flex-start;">
<div style="flex: 1;">
<p style="color: #64748B; font-size: 1rem; margin-bottom: 5px;">Drift Score</p>
<div style="font-size: 2.5rem; font-weight: 700; color: #1E293B;">{drift_score:.2%}</div>
<div style="margin-top: 5px;">
<span style="background-color: {bg_color}; color: {color}; padding: 4px 12px; border-radius: 99px; font-weight: 600; font-size: 0.9rem;">{status}</span>
</div>
</div>
<div style="flex: 2; border-left: 1px solid #E2E8F0; padding-left: 30px;">
<p style="color: #64748B; margin-top: 0; line-height: 1.6;">
Drift score mengukur perubahan distribusi data. Skor tinggi menandakan model mungkin perlu di-retrain.
</p>
<div style="margin-top: 20px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
<span style="font-weight: 600; color: #475569;">Status: <span style="color: {color};">{status}</span></span>
<span style="color: #94A3B8;">{drift_score:.4f}</span>
</div>
<div style="background-color: #F1F5F9; border-radius: 99px; height: 10px; width: 100%; overflow: hidden;">
<div style="background-color: {bar_color}; width: {progress_width}%; height: 100%; border-radius: 99px;"></div>
</div>
</div>
</div>
</div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_prediction_distribution(metrics_summary: Dict[str, Dict[str, Any]]):
    """Render prediction distribution chart per model version."""
    st.markdown("### 🖥️ Frekuensi Prediksi")
    
    if not metrics_summary:
        st.info("Belum ada data distribusi tersedia")
        return
    
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
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=versions,
        y=counts,
        marker=dict(color='#007bff', line=dict(color='white', width=1)),
        text=counts,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Prediksi: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Versi Model",
        yaxis_title="Jumlah Prediksi",
        title="Jumlah Prediksi per Versi Model",
        showlegend=False,
        height=400,
        margin=dict(b=80, t=40, l=20, r=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_monitoring_dashboard(monitoring_service):
    """Main function to render complete monitoring dashboard."""
    try:
        with st.spinner("⏳ Memuat data monitoring..."):
            selected_version = st.session_state.get('selected_model_version')
            dashboard_data = monitoring_service.get_dashboard_data(selected_version)
            metrics_summary = dashboard_data['metrics_summary']
            latency_data = dashboard_data['latency_data']
            drift_score = dashboard_data['drift_score']
        
        # Summary Metrics
        total_predictions = sum(m.get('prediction_count', 0) for m in metrics_summary.values())
        avg_latency_all = sum(latency_data) / len(latency_data) * 1000 if latency_data else 0
        
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Prediksi", total_predictions)
        col2.metric("Rata-rata Latency", f"{avg_latency_all:.2f} ms")
        col3.metric("Drift Score Global", f"{drift_score:.1%}", delta_color="inverse")
        
        st.markdown("---")
        
        # Dashboard sections
        render_drift_score(drift_score)
        st.markdown("---")
        
        render_metrics_table(metrics_summary)
        st.markdown("---")
        
        render_prediction_distribution(metrics_summary)
        st.markdown("---")
        
        render_latency_histogram(latency_data, selected_version)
            
    except ConnectionError as e:
        st.error(f"❌ Gagal terhubung ke database: {str(e)}")
        st.info("💡 Periksa koneksi database dan coba refresh halaman")
        logger.error(f"Database connection error in monitoring: {e}", exc_info=True)
    except Exception as e:
        st.error(f"❌ Gagal memuat dashboard monitoring: {str(e)}")
        st.info("💡 Silakan refresh halaman atau hubungi administrator")
        logger.error(f"Error rendering monitoring dashboard: {e}", exc_info=True)
