"""
Main area components refactored for split-screen layout and modern UI.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from config.settings import settings

def render_input_section() -> str:
    """
    Render clean input section within a card-like visual.
    Returns:
        str: Input text
    """
    # Container for Input "Card"
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown('<div class="card-header">📝 Input Analysis</div>', unsafe_allow_html=True)
    st.caption("Masukkan teks ulasan atau komentar pelanggan untuk dianalisis.")
    
    # Model context
    model_version = st.session_state.get('selected_model_version', 'v1')
    
    # Examples
    examples = {
        'v1': {
            "Positif": "Luar biasa! Produk ini sangat bagus.",
            "Negatif": "Sangat mengecewakan, tidak akan beli lagi."
        },
        'v2': {
            "Positive": "This is amazing, highly recommended!",
            "Negative": "Terrible experience, waste of money."
        }
    }
    
    # Pill-like buttons for examples
    current_examples = examples.get(model_version, examples['v1'])
    cols = st.columns(len(current_examples))
    for i, (label, text) in enumerate(current_examples.items()):
        if cols[i].button(f"📌 {label}", key=f"ex_{i}", use_container_width=True):
             st.session_state['text_input_area'] = text

    # Text Area
    initial_val = st.session_state.get('text_input_area', "")
    text = st.text_area(
        label="Input Text",
        value=initial_val,
        height=200,
        placeholder="Ketik atau tempel teks di sini...",
        key='text_input_area',
        label_visibility="collapsed"
    )
    
    # Char Counter
    if text:
        st.caption(f"{len(text)}/{settings.MAX_INPUT_LENGTH} karakter")
    
    st.markdown('</div>', unsafe_allow_html=True) # End card
    
    return text

def render_prediction_button(enabled: bool = True) -> bool:
    """
    Render primary action button.
    """
    # Using columns to center or stretch
    if st.button("🔍 ANalisis SEKARANG", type="primary", disabled=not enabled, use_container_width=True):
        return True
    return False

def render_results_section(prediction_result: Dict[str, Any]):
    """
    Render results in a card layout. 
    Adapts based on User Mode (Beginner vs Expert).
    """
    if not prediction_result:
        # Empty state
        st.markdown(
            """
            <div class="card" style="text-align: center; color: #9aa0a6; padding: 40px;">
                <h3>👈 Menunggu Input</h3>
                <p>Silakan masukkan teks dan tekan tombol analisis</p>
                <div style="font-size: 3rem; opacity: 0.3;">📊</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Helper function for colors
    def get_color(pred):
        p = pred.lower()
        if p in ['positif', 'positive']: return '#34A853', '😊' # Google Green
        if p in ['negatif', 'negative']: return '#EA4335', '😠' # Google Red
        return '#FBBC04', '😐' # Google Yellow/Neutral

    pred_label = prediction_result.get('prediction', 'Unknown')
    confidence = prediction_result.get('confidence', 0.0)
    color, icon = get_color(pred_label)
    
    user_mode = st.session_state.get('user_mode', 'Beginner')
    
    # Result Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🎯 Hasil Analisis</div>', unsafe_allow_html=True)
    
    # Hero/Highlighted Result
    st.markdown(
        f"""
        <div style="background-color: {color}15; border-left: 5px solid {color}; padding: 20px; border-radius: 4px; margin-bottom: 20px;">
            <h2 style="color: {color}; margin:0; display:flex; align-items:center; gap: 10px;">
                {icon} {pred_label.upper()}
            </h2>
            <p style="margin:0; color: #3c4043; font-weight: 500;">
                Tingkat Keyakinan: {confidence:.0%}
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Additional Metrics for Expert Mode
    if user_mode == "Expert":
        st.markdown("#### 🔬 Technical Metrics")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Latency", f"{prediction_result.get('latency', 0)*1000:.0f} ms")
        with cols[1]:
            st.metric("Model", prediction_result.get('metadata', {}).get('version', 'N/A'))
            
        # Confidence Bar Chart (Simple)
        st.markdown("#### Confidence Distribution")
        _render_simple_bar(confidence, color)
        
        # Raw Data Expander
        with st.expander("📋 View JSON Output"):
            st.json(prediction_result)
            
    st.markdown('</div>', unsafe_allow_html=True) # End card

def _render_simple_bar(value, color):
    """Simple HTML/CSS bar to avoid overhead of Plotly for simple bars."""
    st.markdown(
        f"""
        <div style="background-color: #f1f3f4; border-radius: 4px; height: 8px; width: 100%; margin-top: 5px;">
            <div style="background-color: {color}; width: {value*100}%; height: 100%; border-radius: 4px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #5f6368; margin-top: 4px;">
            <span>0%</span>
            <span>{value:.1%}</span>
            <span>100%</span>
        </div>
        """,
        unsafe_allow_html=True
    )

