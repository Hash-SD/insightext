"""
Test suite untuk verifikasi bug fix di model_updater.py
Bug: NoneType formatting error ketika metrics tidak memiliki 'accuracy' key
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import json
import tempfile
import shutil
from models.model_updater import ModelUpdater, ModelUpdateValidator
from models.model_archiver import ModelArchiver


class TestModelUpdaterBugFix:
    """Test untuk verifikasi bug fix pada model_updater.py"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories untuk testing"""
        base_dir = tempfile.mkdtemp()
        current_model = Path(base_dir) / 'saved_model'
        archive_dir = Path(base_dir) / 'archived'
        current_model.mkdir()
        
        yield {
            'base': base_dir,
            'current_model': str(current_model),
            'archive': str(archive_dir)
        }
        
        # Cleanup
        shutil.rmtree(base_dir)
    
    @pytest.fixture
    def setup_model_files(self, temp_dirs):
        """Setup minimal model files"""
        model_path = Path(temp_dirs['current_model'])
        
        # Create placeholder model files
        (model_path / 'model_pipeline.pkl').touch()
        (model_path / 'preprocessor.pkl').touch()
        
        # Create config dengan metrics yang punya 'accuracy'
        config = {
            'metrics': {
                'accuracy': 0.6972,
                'f1_score': 0.6782
            }
        }
        with open(model_path / 'training_config.json', 'w') as f:
            json.dump(config, f)
        
        yield temp_dirs
    
    def test_update_with_valid_metrics(self, setup_model_files):
        """Test: Update dengan metrics yang valid (dengan 'accuracy' key)"""
        updater = ModelUpdater(
            current_model_path=setup_model_files['current_model'],
            archive_base_path=setup_model_files['archive']
        )
        
        # Setup new model directory
        new_model_dir = Path(setup_model_files['base']) / 'new_model'
        new_model_dir.mkdir()
        (new_model_dir / 'model_pipeline.pkl').touch()
        (new_model_dir / 'preprocessor.pkl').touch()
        (new_model_dir / 'training_config.json').touch()
        
        new_metrics = {
            'accuracy': 0.75,
            'f1_score': 0.73
        }
        
        # Ini harus PASS karena metrics valid
        success, report = updater.update_model_v1(
            new_model_path=str(new_model_dir),
            new_metrics=new_metrics,
            update_reason='Test update with valid metrics',
            auto_validate=False  # Skip validation untuk fokus ke logging bug
        )
        
        assert success is True, f"Update should succeed. Report: {report}"
        assert report['summary']['old_accuracy'] == 0.6972
        assert report['summary']['new_accuracy'] == 0.75
    
    def test_update_with_missing_accuracy_in_current_metrics(self, temp_dirs):
        """
        Test: BUG FIX VERIFICATION
        Update dengan current_metrics yang TIDAK punya 'accuracy' key
        BEFORE FIX: Akan crash dengan TypeError
        AFTER FIX: Harus handle gracefully dengan default value 0.0
        """
        updater = ModelUpdater(
            current_model_path=temp_dirs['current_model'],
            archive_base_path=temp_dirs['archive']
        )
        
        # Setup model dengan config tanpa 'accuracy' key
        model_path = Path(temp_dirs['current_model'])
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / 'model_pipeline.pkl').touch()
        (model_path / 'preprocessor.pkl').touch()
        
        config_without_accuracy = {
            'metrics': {}  # ❌ NO accuracy key!
        }
        with open(model_path / 'training_config.json', 'w') as f:
            json.dump(config_without_accuracy, f)
        
        # Setup new model
        new_model_dir = Path(temp_dirs['base']) / 'new_model'
        new_model_dir.mkdir()
        (new_model_dir / 'model_pipeline.pkl').touch()
        (new_model_dir / 'preprocessor.pkl').touch()
        (new_model_dir / 'training_config.json').touch()
        
        new_metrics = {
            'accuracy': 0.75,
            'f1_score': 0.73
        }
        
        # BEFORE FIX: This would crash with:
        # TypeError: unsupported format string passed to NoneType.__format__
        #
        # AFTER FIX: Should handle gracefully
        try:
            success, report = updater.update_model_v1(
                new_model_path=str(new_model_dir),
                new_metrics=new_metrics,
                update_reason='Test with missing accuracy',
                auto_validate=False
            )
            
            # Seharusnya SUCCESS, bukan crash!
            assert success is True, f"Should succeed even with missing current accuracy. Report: {report}"
            assert report['summary']['old_accuracy'] is None or report['summary']['old_accuracy'] == 0.0
            assert report['summary']['new_accuracy'] == 0.75
            
        except TypeError as e:
            if "unsupported format string passed to NoneType" in str(e):
                pytest.fail(f"❌ BUG NOT FIXED: {e}. Still getting NoneType formatting error!")
            else:
                raise
    
    def test_update_with_missing_both_accuracy_values(self, temp_dirs):
        """
        Test: WORST CASE - Neither current nor new metrics punya 'accuracy'
        Should still handle gracefully
        """
        updater = ModelUpdater(
            current_model_path=temp_dirs['current_model'],
            archive_base_path=temp_dirs['archive']
        )
        
        # Setup current model WITHOUT accuracy
        model_path = Path(temp_dirs['current_model'])
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / 'model_pipeline.pkl').touch()
        (model_path / 'preprocessor.pkl').touch()
        
        with open(model_path / 'training_config.json', 'w') as f:
            json.dump({'metrics': {}}, f)
        
        # Setup new model
        new_model_dir = Path(temp_dirs['base']) / 'new_model'
        new_model_dir.mkdir()
        (new_model_dir / 'model_pipeline.pkl').touch()
        (new_model_dir / 'preprocessor.pkl').touch()
        (new_model_dir / 'training_config.json').touch()
        
        # NEW metrics ALSO tanpa accuracy (validation should fail, but test resilience)
        new_metrics = {}
        
        try:
            success, report = updater.update_model_v1(
                new_model_path=str(new_model_dir),
                new_metrics=new_metrics,
                update_reason='Test with zero metrics',
                auto_validate=False
            )
            
            # Should not crash on logging
            assert report is not None
            
        except TypeError as e:
            if "unsupported format string passed to NoneType" in str(e):
                pytest.fail(f"❌ BUG NOT FIXED: Crash on NoneType formatting: {e}")
            else:
                raise


class TestModelUpdateValidator:
    """Test validator logic"""
    
    def test_validate_model_performance_with_none_values(self):
        """Test validator dengan None metrics"""
        validator = ModelUpdateValidator()
        
        # Test dengan empty metrics
        is_valid, details = validator.validate_model_performance({})
        
        # Should fail gracefully
        assert is_valid is False
        assert 'accuracy_check' in details
        assert 'f1_score_check' in details
        assert details['accuracy_check']['value'] == 0
        assert details['f1_score_check']['value'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
