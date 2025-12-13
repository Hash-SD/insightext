"""
Naive Bayes Model Loader for Sentiment Analysis.
Supports multi-model: v1 (Indonesian) and v2 (IMDB English)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from models.text_preprocessor import TextPreprocessor

# Label mappings
LABEL_MAP_V1 = {"negatif": 0, "netral": 1, "positif": 2}
ID_TO_LABEL_V1 = {v: k for k, v in LABEL_MAP_V1.items()}

LABEL_MAP_V2 = {"negative": 0, "positive": 1}
ID_TO_LABEL_V2 = {v: k for k, v in LABEL_MAP_V2.items()}


class NaiveBayesModelLoader:
    """Model loader for Naive Bayes Sentiment Analysis."""
    
    DEFAULT_METADATA = {
        'v1': {
            'name': 'Naive Bayes Sentiment Analysis',
            'model_type': 'MultinomialNB',
            'vectorizer_type': 'TfidfVectorizer',
            'version': 'v1',
            'task': 'sentiment-analysis',
            'language': 'Indonesian',
            'labels': ['negatif', 'netral', 'positif'],
            'id_to_label': ID_TO_LABEL_V1
        },
        'v2': {
            'name': 'Naive Bayes IMDB Sentiment',
            'model_type': 'MultinomialNB',
            'vectorizer_type': 'TfidfVectorizer',
            'version': 'v2',
            'task': 'sentiment-analysis',
            'language': 'English',
            'labels': ['negative', 'positive'],
            'id_to_label': ID_TO_LABEL_V2
        }
    }
    
    def __init__(self, model_path: Optional[str] = None, version: str = 'v1'):
        self.logger = logging.getLogger(__name__)
        self.version = version
        self._setup_paths(model_path)
        self._init_state()
    
    def _setup_paths(self, model_path: Optional[str]):
        """Setup file paths based on version."""
        if self.version == 'v2':
            self.model_path = Path(model_path) if model_path else Path("models")
            self.model_file = self.model_path / 'naive_bayes_imdb.pkl'
            self.vectorizer_file = self.model_path / 'tfidf_vectorizer_imdb.pkl'
            self.metadata_file = self.model_path / 'model_metadata_imdb.pkl'
        else:
            self.model_path = Path(model_path) if model_path else Path("models/saved_model")
            self.model_file = self.model_path / 'model_pipeline.pkl'
            self.vectorizer_file = None
            self.metadata_file = self.model_path / 'training_config.json'
    
    def _init_state(self):
        """Initialize state variables."""
        self.model = None
        self.vectorizer = None
        self.preprocessor = None
        self.config = None
        self.is_loaded = False
        self.metadata = self.DEFAULT_METADATA.get(self.version, {}).copy()
    
    def load_model(self) -> bool:
        """Load model pipeline and preprocessor."""
        try:
            if self.version == 'v2':
                return self._load_v2_model()
            return self._load_v1_model()
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            self.is_loaded = False
            return False
    
    def _load_v2_model(self) -> bool:
        """Load IMDB English model (v2)."""
        if not self.model_file.exists():
            self.logger.warning(f"IMDB Model not found at {self.model_file}")
            self.preprocessor = TextPreprocessor()
            return False
        
        self.logger.info(f"Loading IMDB model from {self.model_file}")
        with open(self.model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        self.logger.info(f"Loading vectorizer from {self.vectorizer_file}")
        with open(self.vectorizer_file, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                saved_metadata = pickle.load(f)
            self.metadata.update({
                'accuracy': saved_metadata.get('metrics', {}).get('accuracy', 0),
                'f1_score': saved_metadata.get('metrics', {}).get('f1_score', 0),
                'best_params': saved_metadata.get('best_params', {})
            })
        
        self.preprocessor = TextPreprocessor()
        self.is_loaded = True
        self.logger.info(f"Model {self.version} loaded successfully!")
        return True
    
    def _load_v1_model(self) -> bool:
        """Load Indonesian model (v1)."""
        model_file = self.model_path / 'model_pipeline.pkl'
        preprocessor_file = self.model_path / 'preprocessor.pkl'
        config_file = self.model_path / 'training_config.json'
        
        if not model_file.exists():
            self.logger.warning(f"Model not found at {model_file}")
            self.preprocessor = TextPreprocessor()
            return False
        
        self.logger.info(f"Loading model from {model_file}")
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        if preprocessor_file.exists():
            with open(preprocessor_file, 'rb') as f:
                self.preprocessor = pickle.load(f)
        else:
            self.preprocessor = TextPreprocessor()
        
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
    
    def predict(self, text: str, return_all_scores: bool = False) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """Predict sentiment from text."""
        if not self.is_loaded:
            self.logger.info("Model not loaded, loading now...")
            if not self.load_model():
                raise RuntimeError("Failed to load model. Please train the model first.")
        
        try:
            cleaned_text = self.preprocessor.clean_text(text)
            
            if self.version == 'v2' and self.vectorizer is not None:
                return self._predict_v2(cleaned_text, return_all_scores)
            return self._predict_v1(cleaned_text, return_all_scores)
        except Exception as e:
            self.logger.error(f"Prediction error: {e}", exc_info=True)
            raise
    
    def _predict_v2(self, cleaned_text: str, return_all_scores: bool) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """Predict using v2 model."""
        text_tfidf = self.vectorizer.transform([cleaned_text])
        prediction_idx = self.model.predict(text_tfidf)[0]
        proba = self.model.predict_proba(text_tfidf)[0]
        
        prediction = ID_TO_LABEL_V2.get(prediction_idx, str(prediction_idx))
        confidence = proba[prediction_idx]
        
        all_scores = None
        if return_all_scores:
            all_scores = {ID_TO_LABEL_V2.get(i, str(i)): prob for i, prob in enumerate(proba)}
        
        return prediction, confidence, all_scores
    
    def _predict_v1(self, cleaned_text: str, return_all_scores: bool) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """Predict using v1 model."""
        prediction = self.model.predict([cleaned_text])[0]
        proba = self.model.predict_proba([cleaned_text])[0]
        
        classes = self.model.classes_
        pred_idx = list(classes).index(prediction)
        confidence = proba[pred_idx]
        
        all_scores = None
        if return_all_scores:
            all_scores = {cls: prob for cls, prob in zip(classes, proba)}
        
        return prediction, confidence, all_scores
    
    def predict_batch(self, texts: list) -> list:
        """Batch prediction for multiple texts."""
        if not self.is_loaded:
            self.load_model()
        
        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
        
        if self.version == 'v2' and self.vectorizer is not None:
            texts_tfidf = self.vectorizer.transform(cleaned_texts)
            predictions_idx = self.model.predict(texts_tfidf)
            probas = self.model.predict_proba(texts_tfidf)
            
            return [
                (ID_TO_LABEL_V2.get(pred_idx, str(pred_idx)), proba[pred_idx])
                for pred_idx, proba in zip(predictions_idx, probas)
            ]
        
        predictions = self.model.predict(cleaned_texts)
        probas = self.model.predict_proba(cleaned_texts)
        classes = list(self.model.classes_)
        
        return [
            (pred, proba[classes.index(pred)])
            for pred, proba in zip(predictions, probas)
        ]
    
    def get_model_metadata(self) -> Dict[str, Any]:
        return self.metadata.copy()
    
    def get_model_version(self) -> str:
        return self.metadata.get('version', 'v1')
    
    def is_model_loaded(self) -> bool:
        return self.is_loaded


# Singleton instance
_model_instance: Optional[NaiveBayesModelLoader] = None


def get_model_loader(model_path: Optional[str] = None) -> NaiveBayesModelLoader:
    """Get singleton instance of NaiveBayesModelLoader."""
    global _model_instance
    
    if _model_instance is None:
        _model_instance = NaiveBayesModelLoader(model_path=model_path)
        _model_instance.load_model()
    
    return _model_instance


def predict_sentiment(text: str) -> Tuple[str, float]:
    """Convenience function to predict sentiment."""
    loader = get_model_loader()
    prediction, confidence, _ = loader.predict(text)
    return prediction, confidence
