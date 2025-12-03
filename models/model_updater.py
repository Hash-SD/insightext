"""
Model Updater untuk Naive Bayes Sentiment Analysis.

Modul ini menyediakan fungsi untuk:
1. Update model v1 dengan model baru yang sudah dilatih
2. Validasi model baru sebelum deployment
3. Automatic archiving model lama
4. Rollback jika diperlukan
5. Generate change log dan update report
"""

import json
import logging
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from models.model_archiver import ModelArchiver


class ModelUpdateValidator:
    """
    Validator untuk memastikan model baru layak untuk production.
    Melakukan quality checks sebelum model di-deploy.
    """
    
    def __init__(self):
        """Initialize model validator."""
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate_model_structure(self, model_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate struktur file model.
        
        Args:
            model_path: Path ke direktori model
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        try:
            model_dir = Path(model_path)
            required_files = ['model_pipeline.pkl', 'preprocessor.pkl', 'training_config.json']
            
            details = {
                'path_exists': model_dir.exists(),
                'required_files': {}
            }
            
            for file in required_files:
                file_path = model_dir / file
                details['required_files'][file] = file_path.exists()
            
            is_valid = all(details['required_files'].values())
            return is_valid, details
            
        except Exception as e:
            self.logger.error(f"Error validating model structure: {str(e)}")
            return False, {'error': str(e)}
    
    def validate_model_performance(
        self,
        metrics: Dict[str, Any],
        min_accuracy: float = 0.60,
        min_f1_score: float = 0.50
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate performance metrics model.
        
        Args:
            metrics: Dictionary berisi metrics model
            min_accuracy: Minimum accuracy threshold
            min_f1_score: Minimum F1 score threshold
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        details = {
            'accuracy_check': {
                'value': metrics.get('accuracy', 0),
                'threshold': min_accuracy,
                'passed': metrics.get('accuracy', 0) >= min_accuracy
            },
            'f1_score_check': {
                'value': metrics.get('f1_score', 0),
                'threshold': min_f1_score,
                'passed': metrics.get('f1_score', 0) >= min_f1_score
            }
        }
        
        is_valid = details['accuracy_check']['passed'] and details['f1_score_check']['passed']
        return is_valid, details
    
    def validate_prediction_function(
        self,
        predict_func: Callable,
        test_inputs: Optional[list] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate bahwa prediction function berfungsi dengan baik.
        
        Args:
            predict_func: Fungsi prediksi untuk di-test
            test_inputs: List test inputs (optional)
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        details = {
            'function_callable': callable(predict_func),
            'test_results': []
        }
        
        if test_inputs is None:
            test_inputs = [
                "Saya sangat senang dengan produk ini",
                "Ini adalah pengalaman yang buruk",
                "Informasi cukup netral dan faktual"
            ]
        
        try:
            for test_input in test_inputs:
                result = predict_func(test_input)
                details['test_results'].append({
                    'input': test_input,
                    'output': result,
                    'success': result is not None
                })
            
            is_valid = all(r['success'] for r in details['test_results'])
            return is_valid, details
            
        except Exception as e:
            self.logger.error(f"Error validating prediction function: {str(e)}")
            return False, {'error': str(e)}


class ModelUpdater:
    """
    Main class untuk update model v1 dengan model baru.
    Menangani archiving, validation, dan deployment.
    """
    
    def __init__(
        self,
        current_model_path: str = 'models/saved_model',
        archive_base_path: str = 'models/archived'
    ):
        """
        Initialize model updater.
        
        Args:
            current_model_path: Path ke production model
            archive_base_path: Path ke archive directory
        """
        self.logger = logging.getLogger(__name__)
        self.current_model_path = Path(current_model_path)
        self.archiver = ModelArchiver(archive_base_path)
        self.validator = ModelUpdateValidator()
        self.update_log = []
    
    def update_model_v1(
        self,
        new_model_path: str,
        new_metrics: Dict[str, Any],
        update_reason: Optional[str] = None,
        auto_validate: bool = True,
        new_predict_func: Optional[Callable] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Update model v1 dengan model baru.
        
        Process:
        1. Validate model baru
        2. Archive model lama
        3. Copy model baru ke production
        4. Generate update report
        
        Args:
            new_model_path: Path ke directory model baru
            new_metrics: Metrics dari model baru (accuracy, f1_score, dll)
            update_reason: Alasan update (e.g., 'Improved with balanced data')
            auto_validate: Auto-validate model baru sebelum update (default: True)
            new_predict_func: Prediction function dari model baru untuk validation
            
        Returns:
            Tuple of (success, report_dict)
            
        Example:
            >>> updater = ModelUpdater()
            >>> success, report = updater.update_model_v1(
            ...     new_model_path='path/to/new/model',
            ...     new_metrics={'accuracy': 0.75, 'f1_score': 0.73},
            ...     update_reason='Trained with balanced data using oversampling',
            ...     new_predict_func=model.predict
            ... )
            >>> if success:
            ...     print(f"Model updated successfully: {report}")
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("Starting model v1 update process")
            self.logger.info("=" * 80)
            
            report = {
                'success': False,
                'timestamp': datetime.now().isoformat(),
                'update_reason': update_reason or 'Regular model update',
                'steps': {}
            }
            
            # Step 1: Validate model baru jika auto_validate=True
            if auto_validate:
                self.logger.info("Step 1: Validating new model...")
                
                # Validate structure
                struct_valid, struct_details = self.validator.validate_model_structure(new_model_path)
                report['steps']['structure_validation'] = {
                    'passed': struct_valid,
                    'details': struct_details
                }
                
                if not struct_valid:
                    report['error'] = 'Model structure validation failed'
                    self.logger.error(f"Model structure validation failed: {struct_details}")
                    return False, report
                
                # Validate performance
                perf_valid, perf_details = self.validator.validate_model_performance(new_metrics)
                report['steps']['performance_validation'] = {
                    'passed': perf_valid,
                    'details': perf_details
                }
                
                if not perf_valid:
                    report['error'] = 'Model performance validation failed'
                    self.logger.error(f"Model performance validation failed: {perf_details}")
                    return False, report
                
                # Validate prediction function jika provided
                if new_predict_func is not None:
                    pred_valid, pred_details = self.validator.validate_prediction_function(new_predict_func)
                    report['steps']['prediction_validation'] = {
                        'passed': pred_valid,
                        'details': pred_details
                    }
                    
                    if not pred_valid:
                        report['error'] = 'Prediction function validation failed'
                        self.logger.error(f"Prediction function validation failed: {pred_details}")
                        return False, report
                
                self.logger.info("✓ All validations passed")
            
            # Step 2: Get current model metrics untuk archiving
            self.logger.info("Step 2: Preparing to archive current model...")
            current_metrics = self._get_current_model_metrics()
            
            # Step 3: Archive current model
            self.logger.info("Step 3: Archiving current model v1...")
            archive_path = self.archiver.archive_model(
                version='v1',
                current_model_path=str(self.current_model_path),
                metrics=current_metrics,
                notes=f"Archived before update: {update_reason or 'Regular update'}"
            )
            report['steps']['archive'] = {
                'success': True,
                'archive_path': archive_path,
                'archived_metrics': current_metrics
            }
            self.logger.info(f"✓ Current model archived at: {archive_path}")
            
            # Step 4: Deploy new model
            self.logger.info("Step 4: Deploying new model...")
            deploy_success = self._deploy_model(new_model_path)
            report['steps']['deployment'] = {
                'success': deploy_success,
                'new_model_path': str(self.current_model_path)
            }
            
            if not deploy_success:
                report['error'] = 'Model deployment failed'
                self.logger.error("Model deployment failed")
                return False, report
            
            self.logger.info(f"✓ New model deployed successfully")
            
            # Step 5: Generate comparison report
            self.logger.info("Step 5: Generating comparison report...")
            comparison = self.archiver.get_model_comparison(
                current_metrics=new_metrics,
                archive_path=archive_path
            )
            report['steps']['comparison'] = comparison
            report['new_metrics'] = new_metrics
            
            # Step 6: Create update summary
            report['success'] = True
            report['summary'] = {
                'old_model_archive': archive_path,
                'old_accuracy': current_metrics.get('accuracy'),
                'new_accuracy': new_metrics.get('accuracy'),
                'accuracy_improvement': (new_metrics.get('accuracy', 0) - current_metrics.get('accuracy', 0)) * 100,
                'deployment_time': datetime.now().isoformat()
            }
            
            self.logger.info("=" * 80)
            self.logger.info(f"✓ Model update completed successfully")
            # Safe formatting with None defaults
            old_acc = current_metrics.get('accuracy', 0.0)
            new_acc = new_metrics.get('accuracy', 0.0)
            improvement = report['summary']['accuracy_improvement']
            self.logger.info(f"  Old accuracy: {old_acc:.4f}")
            self.logger.info(f"  New accuracy: {new_acc:.4f}")
            self.logger.info(f"  Improvement: {improvement:.2f}%")
            self.logger.info("=" * 80)
            
            # Save update log
            self._save_update_log(report)
            
            return True, report
            
        except Exception as e:
            self.logger.error(f"Error during model update: {str(e)}", exc_info=True)
            return False, {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_current_model_metrics(self) -> Dict[str, Any]:
        """
        Get metrics dari current production model.
        
        Returns:
            Dictionary berisi current model metrics
        """
        try:
            config_path = self.current_model_path / 'training_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config.get('metrics', {})
            return {}
        except Exception as e:
            self.logger.warning(f"Could not read current model metrics: {str(e)}")
            return {}
    
    def _deploy_model(self, source_model_path: str) -> bool:
        """
        Copy model baru ke production directory.
        
        Args:
            source_model_path: Path ke model baru
            
        Returns:
            True jika deploy berhasil
        """
        try:
            source_dir = Path(source_model_path)
            
            # Clear current production model
            if self.current_model_path.exists():
                shutil.rmtree(self.current_model_path)
            
            # Copy new model
            self.current_model_path.mkdir(parents=True, exist_ok=True)
            for file in source_dir.glob('*'):
                if file.is_file():
                    shutil.copy2(file, self.current_model_path / file.name)
            
            self.logger.info(f"Model deployed successfully to: {self.current_model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying model: {str(e)}")
            return False
    
    def _save_update_log(self, report: Dict[str, Any]):
        """
        Save update report ke file log.
        
        Args:
            report: Update report dictionary
        """
        try:
            log_dir = Path('logs/model_updates')
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"update_{timestamp}.json"
            
            with open(log_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Update log saved to: {log_file}")
            
        except Exception as e:
            self.logger.warning(f"Could not save update log: {str(e)}")
    
    def rollback_to_archive(self, archive_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Rollback ke model dari archive jika terjadi masalah.
        
        Args:
            archive_path: Path ke archived model
            
        Returns:
            Tuple of (success, report)
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("Starting rollback process...")
            self.logger.info("=" * 80)
            
            # Archive current (problematic) model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            problematic_archive = self.archiver.archive_model(
                version='v1',
                current_model_path=str(self.current_model_path),
                notes=f"Problematic model archived during rollback at {timestamp}"
            )
            
            # Restore from archive
            success = self.archiver.restore_model(
                archive_path=archive_path,
                restore_to_path=str(self.current_model_path),
                backup_current=False
            )
            
            report = {
                'success': success,
                'timestamp': datetime.now().isoformat(),
                'problematic_model_archive': problematic_archive,
                'restored_from': archive_path
            }
            
            if success:
                self.logger.info("✓ Rollback completed successfully")
            
            return success, report
            
        except Exception as e:
            self.logger.error(f"Error during rollback: {str(e)}", exc_info=True)
            return False, {'error': str(e)}
    
    def list_update_history(self, limit: int = 10) -> list:
        """
        List history update model.
        
        Args:
            limit: Jumlah records yang di-display
            
        Returns:
            List of update records
        """
        try:
            log_dir = Path('logs/model_updates')
            if not log_dir.exists():
                return []
            
            updates = []
            for log_file in sorted(log_dir.glob('update_*.json'), reverse=True)[:limit]:
                with open(log_file, 'r') as f:
                    update_record = json.load(f)
                updates.append({
                    'file': log_file.name,
                    'timestamp': update_record.get('timestamp'),
                    'reason': update_record.get('update_reason'),
                    'success': update_record.get('success')
                })
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Error listing update history: {str(e)}")
            return []
