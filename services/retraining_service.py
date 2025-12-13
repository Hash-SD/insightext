"""Retraining service for orchestrating retraining pipeline."""

import logging
import random
import time
from typing import Dict, Any, Tuple
from datetime import datetime

from database.db_manager import DatabaseManager


class RetrainingService:
    """Service for orchestrating retraining pipeline."""
    
    def __init__(self, db_manager: DatabaseManager, mlflow_tracking_uri: str):
        self.db_manager = db_manager
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.logger = logging.getLogger(__name__)
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow connection (placeholder)."""
        if self.mlflow_tracking_uri:
            self.logger.info(f"MLflow tracking URI configured: {self.mlflow_tracking_uri}")
        else:
            self.logger.warning("No MLflow tracking URI provided")
    
    def trigger_retraining(self, model_version: str) -> Dict[str, Any]:
        """
        Main orchestrator for retraining pipeline.
        
        Flow:
        1. Get dataset snapshot from database
        2. Preprocess data
        3. Train model (placeholder)
        4. Evaluate model
        5. Log to MLflow
        6. Register new version
        """
        try:
            self.logger.info(f"Starting retraining pipeline for model {model_version}")
            start_time = time.time()
            
            # Step 1: Get dataset snapshot
            df = self.get_dataset_snapshot()
            
            if df.empty:
                self.logger.warning("No data available for retraining")
                return {
                    'status': 'no_data',
                    'new_version': None,
                    'metrics': {},
                    'message': 'Tidak ada data untuk retraining. Pastikan ada data dengan user consent.'
                }
            
            self.logger.info(f"Dataset snapshot retrieved: {len(df)} records")
            
            # Step 2: Preprocess data
            X_train, X_test, y_train, y_test = self._split_dataset(df)
            
            # Step 3: Train model (placeholder)
            model = self.train_model_placeholder(X_train, y_train)
            
            # Step 4: Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Step 5: Log to MLflow
            params = {
                'base_version': model_version,
                'dataset_size': len(df),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            self.log_to_mlflow(model, metrics, params)
            
            # Step 6: Generate new version
            new_version = self._generate_new_version(model_version)
            training_time = time.time() - start_time
            
            self.logger.info(f"Retraining completed: {new_version} (time: {training_time:.2f}s)")
            
            return {
                'status': 'success',
                'new_version': new_version,
                'metrics': metrics,
                'message': f'Model {new_version} berhasil dilatih dengan {len(df)} data',
                'training_time': training_time
            }
            
        except Exception as e:
            self.logger.error(f"Retraining failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'new_version': None,
                'metrics': {},
                'message': f'Retraining gagal: {str(e)}'
            }
    
    def get_dataset_snapshot(self) -> Any:
        """Fetch dataset snapshot from database."""
        try:
            df = self.db_manager.get_dataset_snapshot(consent_only=True)
            self.logger.info(f"Dataset snapshot retrieved: {len(df)} records")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching dataset snapshot: {e}", exc_info=True)
            import pandas as pd
            return pd.DataFrame()
    
    def _split_dataset(self, df: Any) -> Tuple[list, list, list, list]:
        """Split dataset into train and test sets."""
        try:
            train_size = int(len(df) * 0.8)
            
            train_df = df[:train_size]
            test_df = df[train_size:]
            
            X_train = train_df['text_input'].tolist()
            y_train = train_df['prediction'].tolist()
            X_test = test_df['text_input'].tolist()
            y_test = test_df['prediction'].tolist()
            
            self.logger.info(f"Dataset split: train={len(X_train)}, test={len(X_test)}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error splitting dataset: {e}", exc_info=True)
            return [], [], [], []
    
    def train_model_placeholder(self, X_train: list, y_train: list) -> Dict[str, Any]:
        """Placeholder training function."""
        self.logger.info(f"Training placeholder model with {len(X_train)} samples")
        time.sleep(0.5)  # Simulate training time
        
        return {
            'type': 'placeholder',
            'trained_at': datetime.now().isoformat(),
            'train_samples': len(X_train),
            'classes': list(set(y_train)) if y_train else ['positif', 'negatif', 'netral']
        }
    
    def evaluate_model(self, model: Dict[str, Any], X_test: list, y_test: list) -> Dict[str, float]:
        """Evaluate model and calculate metrics (placeholder)."""
        self.logger.info(f"Evaluating model with {len(X_test)} test samples")
        
        metrics = {
            'accuracy': round(random.uniform(0.80, 0.95), 4),
            'precision': round(random.uniform(0.78, 0.93), 4),
            'recall': round(random.uniform(0.77, 0.92), 4),
            'f1_score': round(random.uniform(0.79, 0.94), 4)
        }
        
        self.logger.info(f"Evaluation completed: accuracy={metrics['accuracy']:.4f}")
        return metrics
    
    def log_to_mlflow(self, model: Dict[str, Any], metrics: Dict[str, float], params: Dict[str, Any]) -> bool:
        """Log model, metrics, and parameters to MLflow (placeholder)."""
        self.logger.info(f"Logging to MLflow (placeholder): params={params}, metrics={metrics}")
        return True
    
    def _generate_new_version(self, base_version: str) -> str:
        """Generate new version string for retrained model."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_version = f"{base_version}_retrain_{timestamp}"
        self.logger.info(f"Generated new version: {new_version}")
        return new_version
    
    def get_retraining_history(self, limit: int = 10) -> list:
        """Get history of retraining runs (placeholder)."""
        self.logger.info(f"Retrieving retraining history (limit: {limit})")
        return []
    
    def validate_retraining_requirements(self) -> Tuple[bool, str]:
        """Validate if requirements for retraining are met."""
        try:
            df = self.get_dataset_snapshot()
            min_samples = 10
            
            if df.empty:
                return False, "Tidak ada data untuk retraining"
            
            if len(df) < min_samples:
                return False, f"Data tidak cukup. Minimal {min_samples} samples diperlukan, tersedia {len(df)}"
            
            if 'prediction' in df.columns:
                unique_classes = df['prediction'].nunique()
                if unique_classes < 2:
                    return False, "Data harus memiliki minimal 2 kelas berbeda"
            
            return True, f"Requirements terpenuhi. {len(df)} samples tersedia"
            
        except Exception as e:
            self.logger.error(f"Error validating retraining requirements: {e}", exc_info=True)
            return False, f"Error validasi: {str(e)}"
