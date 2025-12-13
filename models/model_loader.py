"""
Model loader for Naive Bayes Sentiment Analysis.
Supports multi-model: v1 (Indonesian) and v2 (IMDB English)
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Tuple

from models.naive_bayes_loader import NaiveBayesModelLoader


class ModelLoader:
    """Model loader for Naive Bayes Sentiment Analysis."""
    
    def __init__(self, mlflow_tracking_uri: Optional[str] = None):
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
    
    def load_model(self, version: str = 'v1', stage: str = 'Production') -> Callable[[str], Tuple[str, float]]:
        """
        Load Naive Bayes model.
        
        Args:
            version: Model version ('v1' = Indonesian, 'v2' = IMDB English)
            stage: Model stage (unused, for compatibility)
            
        Returns:
            Prediction function that accepts text and returns (prediction, confidence)
        """
        self.logger.info(f"Loading Naive Bayes model (version: {version})")
        
        loader = self._get_loader(version)
        if not loader.is_model_loaded():
            loader.load_model()
        
        def predict_func(text: str) -> Tuple[str, float]:
            prediction, confidence, _ = loader.predict(text)
            return prediction, confidence
        
        return predict_func
    
    def get_model_metadata(self, version: str = 'v1') -> Dict[str, Any]:
        """Get model metadata."""
        loader = self._get_loader(version)
        if not loader.is_model_loaded():
            loader.load_model()
        
        metadata = loader.get_model_metadata()
        metadata.update({
            'version': version,
            'stage': 'Production',
            'is_cached': loader.is_model_loaded()
        })
        return metadata
    
    def list_available_versions(self) -> List[str]:
        """List available model versions."""
        return ['v1', 'v2']
    
    def promote_model(self, version: str, from_stage: str, to_stage: str) -> bool:
        """Placeholder for model promotion (not used for single model)."""
        self.logger.info("Model promotion not needed for single Naive Bayes model")
        return True
    
    def clear_cache(self, version: Optional[str] = None):
        """Clear model cache."""
        self.logger.info("Cache clear requested")
        if version and version in self._loaders:
            del self._loaders[version]
        elif version is None:
            self._loaders.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        cached_models = [v for v, loader in self._loaders.items() if loader.is_model_loaded()]
        return {
            'cached_models': cached_models,
            'cache_size': len(cached_models),
            'mlflow_uri': self.mlflow_tracking_uri,
            'default_version': self.default_version,
            'model_type': 'Naive Bayes (MultinomialNB)',
            'vectorizer': 'TF-IDF'
        }
    
    def predict_with_scores(self, text: str, version: str = 'v1') -> Dict[str, Any]:
        """Predict with all scores for each label."""
        loader = self._get_loader(version)
        prediction, confidence, all_scores = loader.predict(text, return_all_scores=True)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'all_scores': all_scores
        }
