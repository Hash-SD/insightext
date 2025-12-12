"""
Module untuk mengelola Custom CSS dan Styling aplikasi.
Mengimplementasikan Google Material Design principles.
"""

import streamlit as st

def load_css():
    """
    Inject custom CSS ke dalam aplikasi Streamlit.
    """
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto:wght@400;500;700&display=swap');

        /* Global Styles */
        .stApp {
            background-color: #FAFAFA;
            font-family: 'Inter', sans-serif;
        }
        
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #202124;
        }
        
        /* Card Component */
        .card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 24px;
            border: 1px solid #E5E7EB;
        }
        
        .card-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a73e8;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Custom Button Styling (Primary) */
        div.stButton > button {
            background-color: #1a73e8;
            color: white;
            border-radius: 8px;
            font-weight: 500;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }
        
        div.stButton > button:hover {
            background-color: #1557b0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transform: translateY(-1px);
        }
        
        /* Secondary/Outline Button */
        div.stButton > button.secondary {
            background-color: white;
            color: #5f6368;
            border: 1px solid #dadce0;
        }
        
        /* Metrics/Results Box */
        .metric-container {
            background-color: #F8F9FA;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            border: 1px solid #E9ECEF;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: #5F6368;
            margin-bottom: 4px;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #202124;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }
        
        /* Input Area Styling */
        .stTextArea textarea {
            border-radius: 8px;
            border-color: #DADCE0;
            font-family: 'Inter', sans-serif;
        }
        
        .stTextArea textarea:focus {
            border-color: #1a73e8;
            box-shadow: 0 0 0 1px #1a73e8;
        }
        
        /* Status/Alert Colors */
        .status-positive { color: #1e8e3e; }
        .status-negative { color: #d93025; }
        .status-neutral { color: #f9ab00; }
        
        /* Utilities */
        .text-small { font-size: 0.875rem; color: #5F6368; }
        .flex-center { display: flex; align-items: center; justify-content: center; }
        .mt-2 { margin-top: 8px; }
        .mb-4 { margin-bottom: 16px; }
        
        </style>
    """, unsafe_allow_html=True)

def card_container(key=None):
    """
    Helper untuk membuat container dengan style card.
    Note: Streamlit container native tidak bisa langsung di-style class-nya dengan mudah 
    tanpa hack JS, jadi kita gunakan approach wrapper atau st.markdown div start/end.
    
    Untuk simplifikasi di python side, kita akan gunakan st.container() biasa 
    tapi content di dalamnya kita wrap dengan HTML div class='card' bila memungkinkan,
    atau kita inject container yang punya background.
    
    Alternative: Just use CSS targeting standard containers if specific classes are hard to inject directly.
    For this 'card' class to work, user needs to wrap content in formatting, e.g.:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    ... elements ...
    st.markdown('</div>', unsafe_allow_html=True)
    """
    pass # Hanya dokumentasi/placeholder, implementasi logic ada di UI components
