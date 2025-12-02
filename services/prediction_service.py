"""
Prediction service untuk orchestrate prediction flow.
Menangani validation, preprocessing, model loading, prediction, dan logging.
"""

import time
import logging
from typing import Dict, Any, Tuple
from database.db_manager import DatabaseManager
from models.model_loader import ModelLoader
from models.text_preprocessor import TextPreprocessor
from utils.validators import validate_text_input, validate_model_version
from utils.privacy import anonymize_pii


class PredictionService:
    """
    Service untuk orchestrate prediction flow dari input validation
    hingga logging ke database.
    """
    
    def __init__(self, db_manager: DatabaseManager, model_loader: ModelLoader):
        """
        Initialize prediction service dengan dependency injection.
        
        Args:
            db_manager: DatabaseManager instance untuk database operations
            model_loader: ModelLoader instance untuk loading models
        """
        self.db_manager = db_manager
        self.model_loader = model_loader
        self.logger = logging.getLogger(__name__)
    
    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Validate input text menggunakan validators utility.
        
        Args:
            text: Input text dari user
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        return validate_text_input(text)
    
    def predict(
        self, 
        text: str, 
        model_version: str, 
        user_consent: bool
    ) -> Dict[str, Any]:
        """
        Main orchestrator untuk prediction flow.
        
        Flow:
        1. Validate input
        2. Preprocess text
        3. Load model
        4. Predict
        5. Measure latency
        6. Log to database (if consent)
        7. Return results
        
        Args:
            text: Input text dari user
            model_version: Versi model yang dipilih (v1-v6)
            user_consent: User consent untuk menyimpan data
            
        Returns:
            Dictionary dengan keys:
                - prediction: str (hasil prediksi)
                - confidence: float (confidence score 0-1)
                - latency: float (waktu prediksi dalam seconds)
                - metadata: dict (informasi tambahan)
                - error: str (jika ada error)
                
        Raises:
            ValueError: Jika input tidak valid
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate input
            self.logger.info(f"Starting prediction with model {model_version}")
            is_valid, error_message = self.validate_input(text)
            
            if not is_valid:
                self.logger.warning(f"Input validation failed: {error_message}")
                return {
                    'prediction': None,
                    'confidence': 0.0,
                    'latency': 0.0,
                    'metadata': {},
                    'error': error_message
                }
            
            # Validate model version
            if not validate_model_version(model_version):
                error_msg = f"Versi model tidak valid: {model_version}"
                self.logger.warning(error_msg)
                return {
                    'prediction': None,
                    'confidence': 0.0,
                    'latency': 0.0,
                    'metadata': {},
                    'error': error_msg
                }
            
            # Step 2: Preprocess text
            self.logger.debug("Preprocessing text")
            preprocessor = TextPreprocessor()
            cleaned_text = preprocessor.preprocess(text)
            
            # Step 3: Load model
            self.logger.debug(f"Loading model {model_version}")
            model_func = self.model_loader.load_model(model_version)
            
            # Step 4: Predict
            self.logger.debug("Running prediction")
            prediction, confidence = model_func(cleaned_text)
            
            # Step 5: Measure latency
            latency = time.time() - start_time
            
            # Step 6: Get model metadata
            metadata = self.model_loader.get_model_metadata(model_version)
            metadata['preprocessed_token_count'] = len(cleaned_text.split())
            
            # Step 7: Log to database (if consent)
            if user_consent:
                try:
                    success = self.log_prediction(
                        text=text,
                        prediction=prediction,
                        confidence=confidence,
                        latency=latency,
                        model_version=model_version,
                        consent=user_consent
                    )
                    if not success:
                        # Add warning to metadata if logging failed
                        metadata['database_warning'] = "Gagal menyimpan ke database"
                except Exception as e:
                    # Log error but don't fail the prediction
                    self.logger.error(
                        f"Failed to log prediction to database: {e}",
                        exc_info=True
                    )
                    # Add warning to metadata
                    metadata['database_warning'] = "Gagal menyimpan ke database"
            else:
                self.logger.info("User consent not given, skipping database logging")
            
            # Step 8: Return results
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'latency': latency,
                'metadata': metadata,
                'error': None
            }
            
            self.logger.info(
                f"Prediction completed: {prediction} "
                f"(confidence: {confidence:.2f}, latency: {latency:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Error saat melakukan prediksi: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'prediction': None,
                'confidence': 0.0,
                'latency': latency,
                'metadata': {},
                'error': error_msg
            }
    
    def log_prediction(
        self,
        text: str,
        prediction: str,
        confidence: float,
        latency: float,
        model_version: str,
        consent: bool
    ) -> bool:
        """
        Log prediction ke database.
        
        Args:
            text: Input text
            prediction: Hasil prediksi
            confidence: Confidence score
            latency: Waktu prediksi
            model_version: Versi model
            consent: User consent
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Check for PII and anonymize if needed
            processed_text, has_pii = anonymize_pii(text)
            
            if has_pii:
                self.logger.info("PII detected and anonymized before saving")
                text_to_save = processed_text
                anonymized = True
            else:
                text_to_save = text
                anonymized = False
            
            # Insert user input
            input_id = self.db_manager.insert_user_input(
                text=text_to_save,
                consent=consent
            )
            
            # Update anonymized flag if needed
            if anonymized:
                # Note: We could add a method to update the anonymized flag
                # For now, we'll handle it in the insert_user_input method
                pass
            
            # Insert prediction
            self.db_manager.insert_prediction(
                input_id=input_id,
                model_version=model_version,
                prediction=prediction,
                confidence=confidence,
                latency=latency
            )
            
            self.logger.info(
                f"Prediction logged successfully: input_id={input_id}, "
                f"anonymized={anonymized}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log prediction: {e}", exc_info=True)
            return False
