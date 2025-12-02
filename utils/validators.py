"""
Input validation utility untuk aplikasi MLOps Streamlit Text AI.
Menyediakan fungsi untuk validasi input teks dan model version.
"""

from typing import Tuple
from config.settings import settings


def validate_text_input(text: str) -> Tuple[bool, str]:
    """
    Validasi input teks berdasarkan panjang minimum dan maksimum.
    
    Args:
        text: Teks input yang akan divalidasi
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
            - is_valid: True jika valid, False jika tidak valid
            - error_message: Pesan error dalam Bahasa Indonesia (kosong jika valid)
    """
    # Check if text is None or not a string
    if text is None:
        return False, "Input teks tidak boleh kosong"
    
    if not isinstance(text, str):
        return False, "Input harus berupa teks"
    
    # Strip whitespace for length validation
    text_stripped = text.strip()
    
    # Check if empty after stripping
    if not text_stripped:
        return False, "Input teks tidak boleh kosong"
    
    # Check minimum length
    if len(text_stripped) < settings.MIN_INPUT_LENGTH:
        return False, f"Input teks minimal {settings.MIN_INPUT_LENGTH} karakter"
    
    # Check maximum length
    if len(text_stripped) > settings.MAX_INPUT_LENGTH:
        return False, f"Input teks maksimal {settings.MAX_INPUT_LENGTH} karakter"
    
    return True, ""


def validate_model_version(version: str) -> bool:
    """
    Validasi versi model apakah termasuk dalam daftar versi yang valid.
    
    Args:
        version: Versi model yang akan divalidasi (contoh: 'v1', 'v2', dll)
    
    Returns:
        bool: True jika versi valid, False jika tidak valid
    """
    if version is None:
        return False
    
    if not isinstance(version, str):
        return False
    
    return version in settings.MODEL_VERSIONS
