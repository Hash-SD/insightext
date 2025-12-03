"""
Model Archiver untuk Naive Bayes Sentiment Analysis.

Modul ini menyediakan fungsi untuk:
1. Mengarsipkan model lama beserta metadata lengkap
2. Mengelola versi model yang di-archive
3. Menampilkan history perubahan model
4. Restore model dari archive jika diperlukan
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class ModelArchiver:
    """
    Model archiver untuk menyimpan dan mengelola versi model lama.
    Setiap model yang di-archive disimpan dengan metadata lengkap.
    """
    
    def __init__(self, archive_base_path: str = 'models/archived'):
        """
        Initialize model archiver.
        
        Args:
            archive_base_path: Path ke direktori archive (default: models/archived)
        """
        self.logger = logging.getLogger(__name__)
        self.archive_base_path = Path(archive_base_path)
        self.archive_base_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"ModelArchiver initialized with archive path: {self.archive_base_path}")
    
    def archive_model(
        self,
        version: str,
        current_model_path: str,
        metrics: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Archive current model dengan metadata lengkap.
        
        Args:
            version: Model version yang di-archive (e.g., 'v1', 'v2')
            current_model_path: Path ke direktori model yang akan di-archive (e.g., 'models/saved_model')
            metrics: Dictionary berisi metrics model (accuracy, f1_score, dll)
            notes: Catatan manual tentang model (optional)
            
        Returns:
            Path ke direktori archive yang baru dibuat
            
        Example:
            >>> archiver = ModelArchiver()
            >>> archive_path = archiver.archive_model(
            ...     version='v1',
            ...     current_model_path='models/saved_model',
            ...     metrics={'accuracy': 0.6972, 'f1_score': 0.6782},
            ...     notes='Original model sebelum update dengan balanced data'
            ... )
        """
        try:
            # Create timestamped archive directory
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
            
            metadata_file = archive_dir / 'archive_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model v{version} archived successfully at: {archive_dir}")
            return str(archive_dir)
            
        except Exception as e:
            self.logger.error(f"Error archiving model: {str(e)}", exc_info=True)
            raise
    
    def list_archived_models(self, version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List semua archived models.
        
        Args:
            version: Filter berdasarkan version (optional)
            
        Returns:
            List of dictionaries berisi info setiap archived model
            
        Example:
            >>> archiver = ModelArchiver()
            >>> archives = archiver.list_archived_models(version='v1')
            >>> for archive in archives:
            ...     print(f"{archive['version']}: {archive['archived_at']}")
        """
        archived_models = []
        
        try:
            for archive_dir in self.archive_base_path.iterdir():
                if archive_dir.is_dir():
                    metadata_file = archive_dir / 'archive_metadata.json'
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Filter by version if specified
                        if version is None or metadata.get('version') == version:
                            metadata['path'] = str(archive_dir)
                            archived_models.append(metadata)
            
            # Sort by archived_at (newest first)
            archived_models.sort(
                key=lambda x: x.get('archived_at', ''),
                reverse=True
            )
            
            return archived_models
            
        except Exception as e:
            self.logger.error(f"Error listing archived models: {str(e)}", exc_info=True)
            return []
    
    def get_archive_info(self, archive_path: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed info tentang specific archived model.
        
        Args:
            archive_path: Path ke archived model directory
            
        Returns:
            Dictionary berisi metadata, atau None jika tidak ada
        """
        try:
            archive_dir = Path(archive_path)
            metadata_file = archive_dir / 'archive_metadata.json'
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Add file listing
                files = [f.name for f in archive_dir.glob('*') if f.is_file()]
                metadata['files'] = files
                metadata['path'] = str(archive_dir)
                
                return metadata
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting archive info: {str(e)}", exc_info=True)
            return None
    
    def restore_model(
        self,
        archive_path: str,
        restore_to_path: str,
        backup_current: bool = True
    ) -> bool:
        """
        Restore model dari archive ke production.
        
        Args:
            archive_path: Path ke archived model
            restore_to_path: Path ke mana model akan di-restore (production path)
            backup_current: Backup model saat ini sebelum restore (default: True)
            
        Returns:
            True jika restore berhasil, False jika gagal
            
        Example:
            >>> archiver = ModelArchiver()
            >>> success = archiver.restore_model(
            ...     archive_path='models/archived/v1_20231201_120000',
            ...     restore_to_path='models/saved_model'
            ... )
        """
        try:
            archive_dir = Path(archive_path)
            restore_dir = Path(restore_to_path)
            
            # Verify archive exists
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
            self.logger.error(f"Error restoring model: {str(e)}", exc_info=True)
            return False
    
    def delete_archive(self, archive_path: str) -> bool:
        """
        Delete specific archived model.
        
        Args:
            archive_path: Path ke archived model yang akan dihapus
            
        Returns:
            True jika delete berhasil, False jika gagal
        """
        try:
            archive_dir = Path(archive_path)
            
            if not archive_dir.exists():
                self.logger.warning(f"Archive directory not found: {archive_path}")
                return False
            
            shutil.rmtree(archive_dir)
            self.logger.info(f"Archive deleted: {archive_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting archive: {str(e)}", exc_info=True)
            return False
    
    def get_model_comparison(
        self,
        current_metrics: Dict[str, Any],
        archive_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Bandingkan metrics model saat ini dengan model di-archive.
        
        Args:
            current_metrics: Metrics dari model production saat ini
            archive_path: Path ke archived model (jika None, ambil yg terbaru)
            
        Returns:
            Dictionary berisi perbandingan metrics
        """
        try:
            # Get archive metrics
            if archive_path is None:
                # Get latest archive
                archives = self.list_archived_models()
                if not archives:
                    return {'error': 'No archived models found'}
                archive_info = archives[0]
            else:
                archive_info = self.get_archive_info(archive_path)
            
            if not archive_info:
                return {'error': 'Archive not found'}
            
            archived_metrics = archive_info.get('metrics', {})
            
            # Calculate differences
            comparison = {
                'current': current_metrics,
                'archived': archived_metrics,
                'differences': {}
            }
            
            for key in set(list(current_metrics.keys()) + list(archived_metrics.keys())):
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
            self.logger.error(f"Error comparing models: {str(e)}", exc_info=True)
            return {'error': str(e)}
