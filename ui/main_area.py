"""
Main area components untuk MLOps Streamlit Text AI application.

Module ini menyediakan fungsi untuk render main content area dengan:
- Text input area
- Prediction button
- Results display dengan confidence chart
- Prediction history table
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from config.settings import settings


def render_text_input() -> str:
    """
    Render text area untuk input teks dengan validation dan example buttons.
    
    Returns:
        str: Input text dari user
    """
    # Get selected model version to determine examples
    model_version = st.session_state.get('selected_model_version', 'v1')
    
    # Example texts based on model version
    if model_version == 'v1':
        # Indonesian examples - dengan sentimen yang sangat jelas
        examples = {
            "Positif": "Luar biasa sekali! Produk ini benar-benar sangat bagus, kualitasnya sempurna, pelayanannya ramah, pengirimannya cepat. Sangat puas dan senang! Sangat merekomendasikan kepada semua orang. Terbaik!",
            "Negatif": "Sangat mengecewakan! Produk jelek, rusak, pelayanan buruk, lambat sekali. Sangat tidak puas dan kecewa. Mengecewakan sekali. Tidak akan beli lagi. Sangat buruk!",
            "Netral": "Paket sudah diterima. Sesuai dengan deskripsi produk. Pengiriman sesuai jadwal. Tidak ada masalah."
        }
    else:
        # English examples (v2 IMDB) - dengan sentimen yang sangat jelas
        examples = {
            "Positive": "This movie is absolutely amazing! I loved every single moment. The plot is brilliant, acting is superb, visuals are stunning. Best movie ever! Highly recommend to everyone!",
            "Negative": "Terrible movie! Worst film I've ever seen. The story is awful, acting is horrible, complete waste of time and money. Very disappointed and angry. Never watching again!",
            "Neutral": "The movie was okay. Some parts were interesting. Nothing special overall. Average experience."
        }
    
    st.markdown("### 📝 Input Teks")
    
    # Example buttons
    st.markdown("Coba dengan contoh:")
    cols = st.columns(len(examples))
    for i, (label, text) in enumerate(examples.items()):
        with cols[i]:
            if st.button(f"Contoh {label}", key=f"btn_ex_{i}", use_container_width=True):
                st.session_state['example_text'] = text
                st.session_state['text_input_area'] = text
    
    # Get initial value
    initial_value = st.session_state.get('example_text', st.session_state.get('text_input_area', ''))
    
    text = st.text_area(
        "Atau ketik teks Anda di sini:",
        value=initial_value,
        height=150,
        max_chars=settings.MAX_INPUT_LENGTH,
        help=(
            f"Masukkan teks minimal {settings.MIN_INPUT_LENGTH} karakter "
            f"dan maksimal {settings.MAX_INPUT_LENGTH} karakter"
        ),
        placeholder="Contoh: Produk ini sangat bagus dan berkualitas tinggi..." if model_version == 'v1' else "Example: This product is great and high quality...",
        key='text_input_area'
    )
    
    # Clear example_text after it's been used
    if 'example_text' in st.session_state:
        del st.session_state['example_text']
    
    # Update session state
    st.session_state['input_text'] = text
    
    # Display character count
    char_count = len(text)
    if char_count > 0:
        color = "green" if char_count >= settings.MIN_INPUT_LENGTH else "red"
        st.markdown(
            f"<small style='color: {color};'>Jumlah karakter: {char_count}/{settings.MAX_INPUT_LENGTH}</small>",
            unsafe_allow_html=True
        )
    
    return text


def render_prediction_button() -> bool:
    """
    Render prediction button.
    
    Returns:
        bool: True jika button di-click
    """
    # Check if input is valid
    text = st.session_state.get('input_text', '')
    is_valid = len(text) >= settings.MIN_INPUT_LENGTH
    
    # Custom CSS untuk tombol biru
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: #1E88E5;
            color: white;
        }
        div.stButton > button:hover {
            background-color: #1565C0;
            color: white;
        }
        div.stButton > button:disabled {
            background-color: #BDBDBD;
            color: #757575;
        }
        </style>
    """, unsafe_allow_html=True)
    
    button_clicked = st.button(
        "🔮 Prediksi",
        disabled=not is_valid,
        use_container_width=True,
        help="Klik untuk melakukan prediksi" if is_valid else f"Masukkan minimal {settings.MIN_INPUT_LENGTH} karakter"
    )
    
    return button_clicked


def render_results(prediction_result: Dict[str, Any]):
    """
    Render hasil prediksi dengan confidence chart dan metadata.
    
    Args:
        prediction_result: Dictionary dengan keys:
            - prediction: str (hasil prediksi)
            - confidence: float (confidence score 0-1)
            - latency: float (waktu prediksi dalam seconds)
            - metadata: dict (model metadata)
            - timestamp: datetime (waktu prediksi)
    """
    if not prediction_result:
        return
    
    st.markdown("### 📊 Hasil Analisis")
    
    # Main result display
    prediction = prediction_result.get('prediction', 'N/A')
    confidence = prediction_result.get('confidence', 0.0)
    
    # Determine color and icon
    pred_lower = prediction.lower()
    
    # Support both Indonesian (v1) and English (v2) labels
    if pred_lower in ['positif', 'positive']:
        color = "green"
        icon = "😊"
        msg = "Sentimen Positif" if pred_lower == 'positif' else "Positive Sentiment"
    elif pred_lower in ['negatif', 'negative']:
        color = "red"
        icon = "😠"
        msg = "Sentimen Negatif" if pred_lower == 'negatif' else "Negative Sentiment"
    elif pred_lower == 'netral':
        color = "orange"
        icon = "😐"
        msg = "Sentimen Netral"
    else:
        color = "gray"
        icon = "❓"
        msg = f"Sentimen: {prediction}"
    
    # Create 3 columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediksi", f"{icon} {prediction.upper()}")
    
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
        
    with col3:
        latency = prediction_result.get('latency', 0.0)
        st.metric("Waktu Proses", f"{latency*1000:.0f} ms")

    # Visual indicator (Colored box)
    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border-radius: 10px;
            background-color: {color};
            color: white;
            text-align: center;
            margin-bottom: 20px;
            opacity: 0.8;
        ">
            <h2 style="margin:0; color: white;">{msg}</h2>
            <p style="margin:0;">Model yakin {confidence:.0%} dengan hasil ini.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Metadata in expander
    _render_metadata_expander(prediction_result)


def _render_confidence_chart(confidence: float, prediction: str):
    """
    Render confidence bar chart menggunakan Plotly.
    
    Args:
        confidence: Confidence score (0-1)
        prediction: Prediction label
    """
    # Determine color based on prediction (support both Indonesian and English)
    color_map = {
        'positif': '#28a745',
        'positive': '#28a745',
        'negatif': '#dc3545',
        'negative': '#dc3545',
        'netral': '#ffc107'
    }
    color = color_map.get(prediction.lower(), '#007bff')
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[confidence],
        y=['Confidence'],
        orientation='h',
        marker=dict(
            color=color,
            line=dict(color=color, width=2)
        ),
        text=[f"{confidence:.1%}"],
        textposition='inside',
        textfont=dict(size=16, color='white'),
        hovertemplate='<b>Confidence:</b> %{x:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis=dict(
            range=[0, 1],
            tickformat='.0%',
            title='Confidence Score'
        ),
        yaxis=dict(showticklabels=False),
        height=150,
        margin=dict(l=0, r=0, t=20, b=40),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, key=f"confidence_chart_{uuid.uuid4().hex[:8]}")


def _render_metadata_expander(prediction_result: Dict[str, Any]):
    """
    Render metadata dalam expander.
    
    Args:
        prediction_result: Prediction result dictionary
    """
    with st.expander("📋 Detail Metadata"):
        metadata = prediction_result.get('metadata', {})
        latency = prediction_result.get('latency', 0.0)
        timestamp = prediction_result.get('timestamp', datetime.now())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Informasi Model:**")
            st.text(f"Versi: {metadata.get('version', 'N/A')}")
            st.text(f"Nama: {metadata.get('name', 'N/A')}")
            st.text(f"Akurasi: {metadata.get('accuracy', 0.0):.1%}")
            st.text(f"F1 Score: {metadata.get('f1_score', 0.0):.1%}")
        
        with col2:
            st.markdown("**Informasi Prediksi:**")
            st.text(f"Latency: {latency*1000:.2f} ms")
            st.text(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            st.text(f"Dataset ID: {metadata.get('dataset_id', 'N/A')}")
            st.text(f"Tokenizer: {metadata.get('tokenizer_version', 'N/A')}")


def render_prediction_history(history_data: List[Dict[str, Any]]):
    """
    Render prediction history table.
    
    Args:
        history_data: List of prediction records dari database
    """
    try:
        if not history_data or len(history_data) == 0:
            st.info("Belum ada riwayat prediksi")
            return
        
        # Prepare data untuk display
        display_data = []
        
        for record in history_data:
            try:
                # Truncate text to 50 characters
                text = record.get('text_input', '')
                truncated_text = text[:50] + '...' if len(text) > 50 else text
                
                # Format timestamp
                timestamp = record.get('timestamp', '')
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        formatted_time = timestamp
                else:
                    formatted_time = str(timestamp)
                
                display_data.append({
                    'Waktu': formatted_time,
                    'Teks': truncated_text,
                    'Model': record.get('model_version', 'N/A'),
                    'Prediksi': record.get('prediction', 'N/A'),
                    'Confidence': f"{record.get('confidence', 0.0):.1%}"
                })
            except Exception as e:
                # Skip invalid records
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Skipping invalid record in history: {e}")
                continue
        
        if not display_data:
            st.warning("⚠️ Tidak dapat menampilkan riwayat prediksi karena format data tidak valid")
            return
        
        # Create DataFrame
        df = pd.DataFrame(display_data)
        
        # Display as dataframe dengan styling
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                'Waktu': st.column_config.TextColumn('Waktu', width='medium'),
                'Teks': st.column_config.TextColumn('Teks', width='large'),
                'Model': st.column_config.TextColumn('Model', width='small'),
                'Prediksi': st.column_config.TextColumn('Prediksi', width='small'),
                'Confidence': st.column_config.TextColumn('Confidence', width='small')
            }
        )
        
        # Display count
        st.caption(f"Menampilkan {len(display_data)} prediksi terakhir")
        
    except Exception as e:
        st.error(f"❌ Gagal menampilkan riwayat prediksi: {str(e)}")
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error rendering prediction history: {e}", exc_info=True)


def render_main_area():
    """
    Main function untuk render complete main content area.
    
    Orchestrates semua main area components:
    - Text input
    
    Note: Actual prediction logic, button handling, results display dan history fetching dilakukan di app.py
    
    Returns:
        str: Input text dari user
    """
    st.markdown("## Prediksi Sentimen Teks")
    st.markdown("Masukkan teks untuk menganalisis sentimen (positif, negatif, atau netral)")
    
    # Text input
    text_input = render_text_input()
    
    return text_input
