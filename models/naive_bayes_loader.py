"""
Naive Bayes Model Loader untuk Sentiment Analysis
Menggunakan model MultinomialNB dengan TF-IDF yang sudah di-train
Mendukung multi-model: v1 (Indonesian) dan v2 (IMDB English)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Import TextPreprocessor from separate module for pickle compatibility
from models.text_preprocessor import TextPreprocessor


# Label mapping for Indonesian model (v1)
LABEL_MAP_V1 = {
    "negatif": 0,
    "netral": 1,
    "positif": 2
}
ID_TO_LABEL_V1 = {v: k for k, v in LABEL_MAP_V1.items()}

# Label mapping for IMDB model (v2)
LABEL_MAP_V2 = {
    "negative": 0,
    "positive": 1
}
ID_TO_LABEL_V2 = {v: k for k, v in LABEL_MAP_V2.items()}


class NaiveBayesModelLoader:
    """
    Model loader untuk Naive Bayes Sentiment Analysis.
    Mendukung multi-model: v1 (Indonesian) dan v2 (IMDB English)
    """
    
    def __init__(self, model_path: Optional[str] = None, version: str = 'v1'):
        """
        Initialize Naive Bayes model loader.
        
        Args:
            model_path: Path ke model yang sudah di-train (default: models/saved_model)
            version: Model version ('v1' = Indonesian, 'v2' = IMDB English)
        """
        self.logger = logging.getLogger(__name__)
        self.version = version
        
        # Set paths based on version
        if version == 'v2':
            self.model_path = Path(model_path) if model_path else Path("models")
            self.model_file = self.model_path / 'naive_bayes_imdb.pkl'
            self.vectorizer_file = self.model_path / 'tfidf_vectorizer_imdb.pkl'
            self.metadata_file = self.model_path / 'model_metadata_imdb.pkl'
        else:
            self.model_path = Path(model_path) if model_path else Path("models/saved_model")
            self.model_file = self.model_path / 'model_pipeline.pkl'
            self.vectorizer_file = None  # Included in pipeline
            self.metadata_file = self.model_path / 'training_config.json'
        
        self.model = None
        self.vectorizer = None
        self.preprocessor = None
        self.config = None
        self.is_loaded = False
        
        # Default metadata based on version
        if version == 'v2':
            self.metadata = {
                'name': 'Naive Bayes IMDB Sentiment',
                'model_type': 'MultinomialNB',
                'vectorizer_type': 'TfidfVectorizer',
                'version': 'v2',
                'task': 'sentiment-analysis',
                'language': 'English',
                'labels': ['negative', 'positive'],
                'id_to_label': ID_TO_LABEL_V2
            }
        else:
            self.metadata = {
                'name': 'Naive Bayes Sentiment Analysis',
                'model_type': 'MultinomialNB',
                'vectorizer_type': 'TfidfVectorizer',
                'version': 'v1',
                'task': 'sentiment-analysis',
                'language': 'Indonesian',
                'labels': ['negatif', 'netral', 'positif'],
                'id_to_label': ID_TO_LABEL_V1
            }
    
    def load_model(self) -> bool:
        """
        Load model pipeline dan preprocessor.
        
        Returns:
            True jika berhasil load model
        """
        try:
            if self.version == 'v2':
                # Load IMDB model (v2)
                if not self.model_file.exists():
                    self.logger.warning(
                        f"IMDB Model not found at {self.model_file}. "
                        f"Please run train_naive_bayes_imdb.py first!"
                    )
                    self.preprocessor = TextPreprocessor()
                    return False
                
                # Load model
                self.logger.info(f"Loading IMDB model from {self.model_file}")
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                # Load vectorizer
                self.logger.info(f"Loading vectorizer from {self.vectorizer_file}")
                with open(self.vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                # Load metadata
                if self.metadata_file.exists():
                    with open(self.metadata_file, 'rb') as f:
                        saved_metadata = pickle.load(f)
                    self.metadata.update({
                        'accuracy': saved_metadata.get('metrics', {}).get('accuracy', 0),
                        'f1_score': saved_metadata.get('metrics', {}).get('f1_score', 0),
                        'best_params': saved_metadata.get('best_params', {})
                    })
                
                self.preprocessor = TextPreprocessor()  # Uses same preprocessor
                
            else:
                # Load Indonesian model (v1)
                model_file = self.model_path / 'model_pipeline.pkl'
                preprocessor_file = self.model_path / 'preprocessor.pkl'
                config_file = self.model_path / 'training_config.json'
                
                if not model_file.exists():
                    self.logger.warning(
                        f"Model not found at {model_file}. "
                        f"Please run train_naive_bayes.py first!"
                    )
                    self.preprocessor = TextPreprocessor()
                    return False
                
                # Load model pipeline
                self.logger.info(f"Loading model from {model_file}")
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                # Load preprocessor
                if preprocessor_file.exists():
                    self.logger.info(f"Loading preprocessor from {preprocessor_file}")
                    with open(preprocessor_file, 'rb') as f:
                        self.preprocessor = pickle.load(f)
                else:
                    self.preprocessor = TextPreprocessor()
                
                # Load config
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        self.config = json.load(f)
                    self.metadata.update({
                        'accuracy': self.config.get('metrics', {}).get('accuracy', 0),
                        'f1_score': self.config.get('metrics', {}).get('f1', 0),
                        'best_params': self.config.get('best_params', {})
                    })
            
            self.is_loaded = True
            self.logger.info(f"Model {self.version} loaded successfully!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            self.is_loaded = False
            return False
    
    def predict(
        self, 
        text: str, 
        return_all_scores: bool = False
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """
        Predict sentiment dari text.
        
        Args:
            text: Input text untuk diprediksi
            return_all_scores: Jika True, return juga semua skor untuk setiap label
            
        Returns:
            Tuple (prediction, confidence, all_scores)
        """
        if not self.is_loaded:
            self.logger.info("Model not loaded, loading now...")
            if not self.load_model():
                raise RuntimeError("Failed to load model. Please train the model first.")
        
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Predict based on version
            if self.version == 'v2' and self.vectorizer is not None:
                # IMDB model: vectorizer is separate
                text_tfidf = self.vectorizer.transform([cleaned_text])
                prediction_idx = self.model.predict(text_tfidf)[0]
                proba = self.model.predict_proba(text_tfidf)[0]
                
                # Convert numeric prediction to label
                prediction = ID_TO_LABEL_V2.get(prediction_idx, str(prediction_idx))
                confidence = proba[prediction_idx]
                
                # All scores
                all_scores = None
                if return_all_scores:
                    all_scores = {ID_TO_LABEL_V2.get(i, str(i)): prob for i, prob in enumerate(proba)}
            else:
                # Indonesian model: pipeline includes vectorizer
                prediction = self.model.predict([cleaned_text])[0]
                proba = self.model.predict_proba([cleaned_text])[0]
                
                # Get confidence
                classes = self.model.classes_
                pred_idx = list(classes).index(prediction)
                confidence = proba[pred_idx]
                
                # All scores
                all_scores = None
                if return_all_scores:
                    all_scores = {cls: prob for cls, prob in zip(classes, proba)}
            
            return prediction, confidence, all_scores
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}", exc_info=True)
            raise
    
    def predict_batch(self, texts: list) -> list:
        """
        Batch prediction untuk multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of (prediction, confidence) tuples
        """
        if not self.is_loaded:
            self.load_model()
        
        # Preprocess all texts
        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
        
        if self.version == 'v2' and self.vectorizer is not None:
            # IMDB model
            texts_tfidf = self.vectorizer.transform(cleaned_texts)
            predictions_idx = self.model.predict(texts_tfidf)
            probas = self.model.predict_proba(texts_tfidf)
            
            results = []
            for pred_idx, proba in zip(predictions_idx, probas):
                prediction = ID_TO_LABEL_V2.get(pred_idx, str(pred_idx))
                confidence = proba[pred_idx]
                results.append((prediction, confidence))
        else:
            # Indonesian model
            predictions = self.model.predict(cleaned_texts)
            probas = self.model.predict_proba(cleaned_texts)
            
            results = []
            classes = list(self.model.classes_)
            for pred, proba in zip(predictions, probas):
                pred_idx = classes.index(pred)
                confidence = proba[pred_idx]
                results.append((pred, confidence))
        
        return results
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self.metadata.copy()
    
    def get_model_version(self) -> str:
        """Get model version."""
        return self.metadata.get('version', 'v1')
    
    def is_model_loaded(self) -> bool:
        """Check apakah model sudah loaded."""
        return self.is_loaded


# Singleton instance
_model_instance: Optional[NaiveBayesModelLoader] = None


def get_model_loader(model_path: Optional[str] = None) -> NaiveBayesModelLoader:
    """
    Get singleton instance dari NaiveBayesModelLoader.
    
    Args:
        model_path: Path ke model (optional)
        
    Returns:
        NaiveBayesModelLoader instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = NaiveBayesModelLoader(model_path=model_path)
        _model_instance.load_model()
    
    return _model_instance


def predict_sentiment(text: str) -> Tuple[str, float]:
    """
    Convenience function untuk predict sentiment.
    
    Args:
        text: Input text
        
    Returns:
        Tuple (prediction, confidence)
    """
    loader = get_model_loader()
    prediction, confidence, _ = loader.predict(text)
    return prediction, confidence
