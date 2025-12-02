"""
Privacy utility untuk aplikasi MLOps Streamlit Text AI.
Menyediakan fungsi untuk deteksi dan anonymisasi PII (Personally Identifiable Information).
"""

import re
from typing import Tuple, List


def anonymize_pii(text: str) -> Tuple[str, bool]:
    """
    Anonymisasi PII dalam teks dengan mengganti email dan nomor telepon dengan placeholder.
    
    Args:
        text: Teks yang akan di-anonymisasi
    
    Returns:
        Tuple[str, bool]: (anonymized_text, has_pii)
            - anonymized_text: Teks yang sudah di-anonymisasi
            - has_pii: True jika ditemukan PII, False jika tidak
    """
    if not text or not isinstance(text, str):
        return text, False
    
    anonymized_text = text
    has_pii = False
    
    # Regex pattern untuk email
    # Matches: user@example.com, user.name@example.co.id, etc.
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, anonymized_text):
        anonymized_text = re.sub(email_pattern, '[EMAIL]', anonymized_text)
        has_pii = True
    
    # Regex pattern untuk nomor telepon Indonesia
    # Matches: 08123456789, +628123456789, 021-12345678, (021) 12345678, etc.
    phone_patterns = [
        r'\+?62\s?8\d{2}[\s-]?\d{3,4}[\s-]?\d{3,4}',  # +62812-3456-7890
        r'0\d{2,3}[\s-]?\d{3,4}[\s-]?\d{3,4}',        # 021-1234-5678 or 0812-3456-7890
        r'\(\d{2,3}\)\s?\d{3,4}[\s-]?\d{3,4}',        # (021) 1234-5678
    ]
    
    for pattern in phone_patterns:
        if re.search(pattern, anonymized_text):
            anonymized_text = re.sub(pattern, '[PHONE]', anonymized_text)
            has_pii = True
    
    return anonymized_text, has_pii


def detect_pii(text: str) -> List[str]:
    """
    Deteksi jenis-jenis PII yang ada dalam teks.
    
    Args:
        text: Teks yang akan dideteksi
    
    Returns:
        List[str]: List berisi jenis PII yang ditemukan (contoh: ['email', 'phone'])
    """
    if not text or not isinstance(text, str):
        return []
    
    pii_types = []
    
    # Check untuk email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, text):
        pii_types.append('email')
    
    # Check untuk nomor telepon
    phone_patterns = [
        r'\+?62\s?8\d{2}[\s-]?\d{3,4}[\s-]?\d{3,4}',
        r'0\d{2,3}[\s-]?\d{3,4}[\s-]?\d{3,4}',
        r'\(\d{2,3}\)\s?\d{3,4}[\s-]?\d{3,4}',
    ]
    
    for pattern in phone_patterns:
        if re.search(pattern, text):
            pii_types.append('phone')
            break  # Only add 'phone' once
    
    return pii_types
