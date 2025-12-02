"""
Model loader untuk Naive Bayes Sentiment Analysis.
Menggunakan model MultinomialNB dengan TF-IDF yang sudah di-train.
Mendukung multi-model: v1 (Indonesian) dan v2 (IMDB English)
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Tuple
from models.naive_bayes_loader import NaiveBayesModelLoader, LABEL_MAP_V1, ID_TO_LABEL_V1, LABEL_MAP_V2, ID_TO_LABEL_V2


class ModelLoader:
    """
    Model loader untuk Naive Bayes Sentiment Analysis.
    Mendukung multi-model: v1 (Indonesian) dan v2 (IMDB English)
    """
    
    def __init__(self, mlflow_tracking_uri: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            mlflow_tracking_uri: URI untuk MLflow (untuk kompatibilitas, tidak digunakan)
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.logger = logging.getLogger(__name__)
        self._loaders: Dict[str, NaiveBayesModelLoader] = {}
        self.default_version = 'v1'
        
        self.logger.info("ModelLoader initialized with Naive Bayes backend (multi-model)")
    
    def _get_loader(self, version: str) -> NaiveBayesModelLoader:
        """Get or create loader for specific version."""
        if version not in self._loaders:
            self._loaders[version] = NaiveBayesModelLoader(version=version)
        return self._loaders[version]
    
    def load_model(
        self, 
        version: str = 'v1', 
        stage: str = 'Production'
    ) -> Callable[[str], Tuple[str, float]]:
        """
        Load Naive Bayes model.
        
        Args:
            version: Model version ('v1' = Indonesian, 'v2' = IMDB English)
            stage: Model stage (tidak digunakan, untuk kompatibilitas)
            
        Returns:
            Prediction function yang menerima text dan return (prediction, confidence)
        """
        self.logger.info(f"Loading Naive Bayes model (version: {version})")
        
        # Get loader for version
        loader = self._get_loader(version)
        
        # Ensure model is loaded
        if not loader.is_model_loaded():
            loader.load_model()
        
        # Return prediction function
        def predict_func(text: str) -> Tuple[str, float]:
            prediction, confidence, _ = loader.predict(text)
            return prediction, confidence
        
        return predict_func
    
    def get_model_metadata(self, version: str = 'v1') -> Dict[str, Any]:
        """
        Get model metadata.
        
        Args:
            version: Model version ('v1' atau 'v2')
            
        Returns:
            Dictionary dengan model metadata
        """
        loader = self._get_loader(version)
        if not loader.is_model_loaded():
            loader.load_model()
        
        metadata = loader.get_model_metadata()
        metadata['version'] = version
        metadata['stage'] = 'Production'
        metadata['is_cached'] = loader.is_model_loaded()
        
        return metadata
    
    def list_available_versions(self) -> List[str]:
        """
        List available model versions.
        
        Returns:
            List of available versions
        """
        return ['v1', 'v2']
    
    def promote_model(
        self, 
        version: str, 
        from_stage: str, 
        to_stage: str
    ) -> bool:
        """
        Placeholder untuk promote model.
        Tidak digunakan karena hanya ada satu model.
        
        Returns:
            True (selalu berhasil untuk kompatibilitas)
        """
        self.logger.info(f"Model promotion not needed for single Naive Bayes model")
        return True
    
    def clear_cache(self, version: Optional[str] = None):
        """
        Clear model cache (untuk kompatibilitas).
        """
        self.logger.info("Cache clear requested")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.
        
        Returns:
            Dictionary dengan cache info
        """
        return {
            'cached_models': ['naive-bayes-tfidf'] if self.nb_loader.is_model_loaded() else [],
            'cache_size': 1 if self.nb_loader.is_model_loaded() else 0,
            'mlflow_uri': self.mlflow_tracking_uri,
            'default_version': self.default_version,
            'model_type': 'Naive Bayes (MultinomialNB)',
            'vectorizer': 'TF-IDF'
        }
    
    def predict_with_scores(
        self, 
        text: str
    ) -> Dict[str, Any]:
        """
        Predict dengan semua scores untuk setiap label.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary dengan prediction, confidence, dan all_scores
        """
        prediction, confidence, all_scores = self.nb_loader.predict(
            text, 
            return_all_scores=True
        )
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'all_scores': all_scores
        }
