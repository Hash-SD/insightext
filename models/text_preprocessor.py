"""
Text Preprocessor untuk Naive Bayes Sentiment Analysis
Modul terpisah agar bisa di-pickle dan di-unpickle dengan benar.
"""

import re
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Preprocessing teks untuk sentiment analysis Bahasa Indonesia.
    Dikembangkan dengan normalisasi slang dan stopwords removal.
    """
    
    def __init__(self):
        # Slang/informal words normalization
        self.slang_dict = {
            # Negasi
            'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
            'gk': 'tidak', 'tdk': 'tidak', 'kagak': 'tidak', 'kaga': 'tidak',
            # Kata ganti
            'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'u/': 'untuk',
            'krn': 'karena', 'karna': 'karena', 'krna': 'karena',
            'tp': 'tapi', 'tpi': 'tapi',
            'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah',
            'blm': 'belum', 'blum': 'belum',
            'sm': 'sama', 'sma': 'sama',
            'aja': 'saja', 'aj': 'saja', 'doang': 'saja',
            'bgt': 'banget', 'bngt': 'banget', 'bener': 'benar', 'bnr': 'benar',
            'emg': 'memang', 'emang': 'memang', 'mmg': 'memang',
            'jg': 'juga', 'jga': 'juga',
            'jd': 'jadi', 'jdi': 'jadi',
            'kalo': 'kalau', 'klo': 'kalau', 'kl': 'kalau',
            'bs': 'bisa', 'bsa': 'bisa',
            'gmn': 'bagaimana', 'gimana': 'bagaimana', 'gmana': 'bagaimana',
            'knp': 'kenapa', 'knapa': 'kenapa',
            # Kata ganti orang
            'sy': 'saya', 'gw': 'saya', 'gue': 'saya', 'gua': 'saya',
            'ak': 'aku', 'aq': 'aku', 'w': 'saya',
            'km': 'kamu', 'kmu': 'kamu', 'lu': 'kamu', 'lo': 'kamu',
            'org': 'orang', 'orng': 'orang', 'org2': 'orang-orang',
            # Kata kerja/aktivitas
            'mkn': 'makan', 'mkanan': 'makanan',
            'ank': 'anak', 'ank2': 'anak-anak',
            'sek': 'sekolah', 'sklh': 'sekolah',
            'skrg': 'sekarang', 'skrang': 'sekarang',
            # Kata sifat
            'bgus': 'bagus', 'bgs': 'bagus',
            'jlk': 'jelek', 'jlek': 'jelek',
            'mantap': 'mantap', 'mantab': 'mantap', 'mantul': 'mantap',
            'keren': 'keren', 'krn': 'keren',
            'parah': 'parah', 'prh': 'parah',
            'ancur': 'hancur', 'hancr': 'hancur',
            # Konteks MBG
            'mbg': 'makan bergizi', 'prabowo': 'prabowo',
            'gratis': 'gratis', 'grtis': 'gratis',
            # Ekspresi
            'wkwk': 'tertawa', 'wkwkwk': 'tertawa', 'haha': 'tertawa',
            'hahaha': 'tertawa', 'xixi': 'tertawa', 'kwkw': 'tertawa',
            'anjir': 'kaget', 'anjay': 'kaget', 'asw': 'kaget',
            'btw': 'ngomong-ngomong',
        }
        
        # Indonesian stopwords (kata yang tidak penting untuk sentiment)
        self.stopwords = set([
            'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
            'pada', 'adalah', 'sebagai', 'dalam', 'tidak', 'akan', 'atau',
            'juga', 'ada', 'mereka', 'ia', 'oleh', 'saya', 'kita', 'kami',
            'anda', 'tersebut', 'dapat', 'sudah', 'telah', 'bisa', 'hanya',
            'seperti', 'lebih', 'antara', 'jadi', 'setelah', 'karena',
            'saat', 'serta', 'sehingga', 'maka', 'tentang', 'ketika',
            'namun', 'bila', 'pun', 'lagi', 'sangat', 'begitu', 'bahwa',
        ])
        
        # Emoticon patterns untuk sentiment
        self.positive_emoticons = [':)', ':-)', ':D', ':-D', 'xD', 'XD', ';)', 
                                    '^^', '<3', ':*', '😊', '😄', '😃', '👍', '❤️']
        self.negative_emoticons = [':(', ':-(', ":'(", 'T_T', 'T.T', '-_-',
                                    '😢', '😭', '😠', '😡', '👎', '💔']
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess single text (alias for clean_text).
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        return self.clean_text(text)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle emoticons - convert to sentiment words
        for emo in self.positive_emoticons:
            if emo in text:
                text = text.replace(emo, ' senang ')
        for emo in self.negative_emoticons:
            if emo in text:
                text = text.replace(emo, ' sedih ')
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters (keep only alphanumeric and space)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Normalize repeated characters (e.g., "baguuuus" -> "bagus")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Normalize slang words
        words = text.split()
        normalized_words = []
        for word in words:
            # Check slang dict
            normalized_word = self.slang_dict.get(word, word)
            normalized_words.append(normalized_word)
        
        text = ' '.join(normalized_words)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_dataframe(self, df, text_column='text'):
        """
        Preprocess entire dataframe.
        
        Args:
            df: pandas DataFrame
            text_column: column name containing text
            
        Returns:
            DataFrame with 'text_cleaned' column added
        """
        logger.info("Preprocessing texts...")
        df['text_cleaned'] = df[text_column].apply(self.clean_text)
        
        # Log some examples
        logger.info("Sample preprocessing results:")
        for i in range(min(3, len(df))):
            orig = str(df[text_column].iloc[i])[:60]
            clean = str(df['text_cleaned'].iloc[i])[:60]
            logger.info(f"  Original: {orig}...")
            logger.info(f"  Cleaned:  {clean}...")
        
        return df
