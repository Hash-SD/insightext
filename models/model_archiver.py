"""
Model Archiver for Naive Bayes Sentiment Analysis.

Provides functionality for:
1. Archiving old models with complete metadata
2. Managing archived model versions
3. Displaying model change history
4. Restoring models from archive
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class ModelArchiver:
    """Model archiver for storing and managing old model versions."""
    
    def __init__(self, archive_base_path: str = 'models/archived'):
        self.logger = logging.getLogger(__name__)
        self.archive_base_path = Path(archive_base_path)
        self.archive_base_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"ModelArchiver initialized: {self.archive_base_path}")
    
    def archive_model(
        self,
        version: str,
        current_model_path: str,
        metrics: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Archive current model with complete metadata.
        
        Returns:
            Path to the newly created archive directory
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = self.archive_base_path / f"{version}_{timestamp}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            source_path = Path(current_model_path)
            if source_path.exists():
                for file in source_path.glob('*'):
                    if file.is_file():
                        shutil.copy2(file, archive_dir / file.name)
                self.logger.info(f"Model files copied to archive: {archive_dir}")
            else:
                self.logger.warning(f"Model source path not found: {source_path}")
            
            # Create metadata file
            metadata = {
                'version': version,
                'archived_at': datetime.now().isoformat(),
                'timestamp': timestamp,
                'notes': notes or 'Model archived during update process',
                'metrics': metrics or {}
            }
            
            with open(archive_dir / 'archive_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model v{version} archived at: {archive_dir}")
            return str(archive_dir)
            
        except Exception as e:
            self.logger.error(f"Error archiving model: {e}", exc_info=True)
            raise
    
    def list_archived_models(self, version: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all archived models, optionally filtered by version."""
        archived_models = []
        
        try:
            for archive_dir in self.archive_base_path.iterdir():
                if not archive_dir.is_dir():
                    continue
                    
                metadata_file = archive_dir / 'archive_metadata.json'
                if not metadata_file.exists():
                    continue
                    
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if version is None or metadata.get('version') == version:
                    metadata['path'] = str(archive_dir)
                    archived_models.append(metadata)
            
            # Sort by archived_at (newest first)
            archived_models.sort(key=lambda x: x.get('archived_at', ''), reverse=True)
            return archived_models
            
        except Exception as e:
            self.logger.error(f"Error listing archived models: {e}", exc_info=True)
            return []
    
    def get_archive_info(self, archive_path: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific archived model."""
        try:
            archive_dir = Path(archive_path)
            metadata_file = archive_dir / 'archive_metadata.json'
            
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['files'] = [f.name for f in archive_dir.glob('*') if f.is_file()]
            metadata['path'] = str(archive_dir)
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error getting archive info: {e}", exc_info=True)
            return None
    
    def restore_model(
        self,
        archive_path: str,
        restore_to_path: str,
        backup_current: bool = True
    ) -> bool:
        """Restore model from archive to production."""
        try:
            archive_dir = Path(archive_path)
            restore_dir = Path(restore_to_path)
            
            if not archive_dir.exists():
                self.logger.error(f"Archive directory not found: {archive_path}")
                return False
            
            # Backup current model if requested
            if backup_current and restore_dir.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.archive_base_path / f"backup_{timestamp}"
                shutil.copytree(restore_dir, backup_path)
                self.logger.info(f"Current model backed up to: {backup_path}")
            
            # Clear restore directory
            if restore_dir.exists():
                shutil.rmtree(restore_dir)
            
            # Restore files
            restore_dir.mkdir(parents=True, exist_ok=True)
            for file in archive_dir.glob('*'):
                if file.is_file() and file.name != 'archive_metadata.json':
                    shutil.copy2(file, restore_dir / file.name)
            
            self.logger.info(f"Model restored from archive: {restore_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring model: {e}", exc_info=True)
            return False
    
    def delete_archive(self, archive_path: str) -> bool:
        """Delete specific archived model."""
        try:
            archive_dir = Path(archive_path)
            
            if not archive_dir.exists():
                self.logger.warning(f"Archive directory not found: {archive_path}")
                return False
            
            shutil.rmtree(archive_dir)
            self.logger.info(f"Archive deleted: {archive_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting archive: {e}", exc_info=True)
            return False
    
    def get_model_comparison(
        self,
        current_metrics: Dict[str, Any],
        archive_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare current model metrics with archived model."""
        try:
            if archive_path is None:
                archives = self.list_archived_models()
                if not archives:
                    return {'error': 'No archived models found'}
                archive_info = archives[0]
            else:
                archive_info = self.get_archive_info(archive_path)
            
            if not archive_info:
                return {'error': 'Archive not found'}
            
            archived_metrics = archive_info.get('metrics', {})
            
            comparison = {
                'current': current_metrics,
                'archived': archived_metrics,
                'differences': {}
            }
            
            all_keys = set(list(current_metrics.keys()) + list(archived_metrics.keys()))
            for key in all_keys:
                current_val = current_metrics.get(key)
                archived_val = archived_metrics.get(key)
                
                if current_val is not None and archived_val is not None:
                    if isinstance(current_val, (int, float)):
                        diff = current_val - archived_val
                        percent_change = (diff / archived_val * 100) if archived_val != 0 else 0
                        comparison['differences'][key] = {
                            'value': diff,
                            'percent': percent_change,
                            'improvement': diff > 0
                        }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}", exc_info=True)
            return {'error': str(e)}
