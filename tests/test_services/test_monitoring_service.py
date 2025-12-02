"""
Unit tests untuk MonitoringService
Tests: metrics aggregation, latency distribution, drift detection
"""

import pytest
from unittest.mock import Mock
from services.monitoring_service import MonitoringService


@pytest.fixture
def mock_db_manager():
    """Create mock DatabaseManager"""
    mock = Mock()
    
    # Mock metrics data
    mock.get_metrics_by_version.return_value = {
        'v1': {
            'prediction_count': 100,
            'avg_confidence': 0.85,
            'avg_latency': 0.234,
            'min_latency': 0.1,
            'max_latency': 0.5
        },
        'v2': {
            'prediction_count': 50,
            'avg_confidence': 0.90,
            'avg_latency': 0.345,
            'min_latency': 0.2,
            'max_latency': 0.6
        }
    }
    
    # Mock query results
    mock.execute_query.return_value = [
        {'latency': 0.1},
        {'latency': 0.2},
        {'latency': 0.3}
    ]
    
    return mock


@pytest.fixture
def monitoring_service(mock_db_manager):
    """Create MonitoringService instance dengan mocked dependencies"""
    return MonitoringService(mock_db_manager)


class TestMetricsSummary:
    """Test metrics summary retrieval"""
    
    def test_get_metrics_summary_success(self, monitoring_service, mock_db_manager):
        """Test successful metrics summary retrieval"""
        metrics = monitoring_service.get_metrics_summary()
        
        assert len(metrics) == 2
        assert 'v1' in metrics
        assert 'v2' in metrics
        assert metrics['v1']['prediction_count'] == 100
        assert metrics['v1']['avg_confidence'] == 0.85
        
        mock_db_manager.get_metrics_by_version.assert_called_once()
    
    def test_get_metrics_summary_empty(self, monitoring_service, mock_db_manager):
        """Test metrics summary dengan no data"""
        mock_db_manager.get_metrics_by_version.return_value = {}
        
        metrics = monitoring_service.get_metrics_summary()
        
        assert metrics == {}
    
    def test_get_metrics_summary_error(self, monitoring_service, mock_db_manager):
        """Test metrics summary dengan database error"""
        mock_db_manager.get_metrics_by_version.side_effect = Exception("DB error")
        
        metrics = monitoring_service.get_metrics_summary()
        
        assert metrics == {}


class TestLatencyDistribution:
    """Test latency distribution retrieval"""
    
    def test_get_latency_distribution_all_versions(self, monitoring_service, mock_db_manager):
        """Test latency distribution untuk all versions"""
        latencies = monitoring_service.get_latency_distribution()
        
        assert len(latencies) == 3
        assert all(isinstance(l, float) for l in latencies)
        
        mock_db_manager.execute_query.assert_called_once()
    
    def test_get_latency_distribution_specific_version(self, monitoring_service, mock_db_manager):
        """Test latency distribution untuk specific version"""
        latencies = monitoring_service.get_latency_distribution(model_version="v1")
        
        assert len(latencies) == 3
        
        # Verify query was called with model_version parameter
        call_args = mock_db_manager.execute_query.call_args
        assert call_args[0][1] == ("v1",)
    
    def test_get_latency_distribution_empty(self, monitoring_service, mock_db_manager):
        """Test latency distribution dengan no data"""
        mock_db_manager.execute_query.return_value = []
        
        latencies = monitoring_service.get_latency_distribution()
        
        assert latencies == []
    
    def test_get_latency_distribution_error(self, monitoring_service, mock_db_manager):
        """Test latency distribution dengan error"""
        mock_db_manager.execute_query.side_effect = Exception("Query error")
        
        latencies = monitoring_service.get_latency_distribution()
        
        assert latencies == []


class TestDriftDetection:
    """Test drift score calculation"""
    
    def test_calculate_drift_score(self, monitoring_service):
        """Test drift score calculation"""
        drift_score = monitoring_service.calculate_drift_score()
        
        assert isinstance(drift_score, float)
        assert 0.0 <= drift_score <= 1.0
    
    def test_calculate_drift_score_multiple_calls(self, monitoring_service):
        """Test drift score calculation consistency"""
        scores = [monitoring_service.calculate_drift_score() for _ in range(5)]
        
        # All scores should be valid
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestPredictionCounts:
    """Test prediction counts retrieval"""
    
    def test_get_prediction_counts_success(self, monitoring_service, mock_db_manager):
        """Test successful prediction counts retrieval"""
        mock_db_manager.execute_query.return_value = [
            {'model_version': 'v1', 'count': 100},
            {'model_version': 'v2', 'count': 50}
        ]
        
        counts = monitoring_service.get_prediction_counts()
        
        assert len(counts) == 2
        assert counts['v1'] == 100
        assert counts['v2'] == 50
    
    def test_get_prediction_counts_empty(self, monitoring_service, mock_db_manager):
        """Test prediction counts dengan no data"""
        mock_db_manager.execute_query.return_value = []
        
        counts = monitoring_service.get_prediction_counts()
        
        assert counts == {}
    
    def test_get_prediction_counts_error(self, monitoring_service, mock_db_manager):
        """Test prediction counts dengan error"""
        mock_db_manager.execute_query.side_effect = Exception("Query error")
        
        counts = monitoring_service.get_prediction_counts()
        
        assert counts == {}


class TestConfidenceDistribution:
    """Test confidence distribution retrieval"""
    
    def test_get_confidence_distribution_all_versions(self, monitoring_service, mock_db_manager):
        """Test confidence distribution untuk all versions"""
        mock_db_manager.execute_query.return_value = [
            {'confidence': 0.85},
            {'confidence': 0.90},
            {'confidence': 0.75}
        ]
        
        confidences = monitoring_service.get_confidence_distribution()
        
        assert len(confidences) == 3
        assert all(isinstance(c, float) for c in confidences)
    
    def test_get_confidence_distribution_specific_version(self, monitoring_service, mock_db_manager):
        """Test confidence distribution untuk specific version"""
        mock_db_manager.execute_query.return_value = [
            {'confidence': 0.85}
        ]
        
        confidences = monitoring_service.get_confidence_distribution(model_version="v1")
        
        assert len(confidences) == 1
        assert confidences[0] == 0.85


class TestPredictionTimeline:
    """Test prediction timeline retrieval"""
    
    def test_get_prediction_timeline_success(self, monitoring_service, mock_db_manager):
        """Test successful timeline retrieval"""
        mock_db_manager.execute_query.return_value = [
            {
                'timestamp': '2023-11-26 10:00:00',
                'model_version': 'v1',
                'confidence': 0.85,
                'latency': 0.234
            }
        ]
        
        timeline = monitoring_service.get_prediction_timeline(limit=100)
        
        assert len(timeline) == 1
        assert 'timestamp' in timeline[0]
        assert 'model_version' in timeline[0]
    
    def test_get_prediction_timeline_with_limit(self, monitoring_service, mock_db_manager):
        """Test timeline retrieval dengan custom limit"""
        monitoring_service.get_prediction_timeline(limit=50)
        
        call_args = mock_db_manager.execute_query.call_args
        assert call_args[0][1] == (50,)


class TestModelComparison:
    """Test model comparison functionality"""
    
    def test_get_model_comparison_success(self, monitoring_service, mock_db_manager):
        """Test successful model comparison"""
        mock_db_manager.execute_query.return_value = [
            {'model_version': 'v1', 'count': 100},
            {'model_version': 'v2', 'count': 50}
        ]
        
        comparison = monitoring_service.get_model_comparison()
        
        assert len(comparison) == 2
        assert 'v1' in comparison
        assert 'total_predictions' in comparison['v1']
        assert comparison['v1']['total_predictions'] == 100
    
    def test_get_model_comparison_empty(self, monitoring_service, mock_db_manager):
        """Test model comparison dengan no data"""
        mock_db_manager.get_metrics_by_version.return_value = {}
        mock_db_manager.execute_query.return_value = []
        
        comparison = monitoring_service.get_model_comparison()
        
        assert comparison == {}


class TestRecentActivity:
    """Test recent activity statistics"""
    
    def test_get_recent_activity_success(self, monitoring_service, mock_db_manager):
        """Test successful recent activity retrieval"""
        mock_db_manager.execute_query.return_value = [
            {
                'total_predictions': 50,
                'avg_confidence': 0.85,
                'avg_latency': 0.234,
                'models_used': 3
            }
        ]
        
        activity = monitoring_service.get_recent_activity(hours=24)
        
        assert activity['total_predictions'] == 50
        assert activity['avg_confidence'] == 0.85
        assert activity['models_used'] == 3
    
    def test_get_recent_activity_no_data(self, monitoring_service, mock_db_manager):
        """Test recent activity dengan no data"""
        mock_db_manager.execute_query.return_value = []
        
        activity = monitoring_service.get_recent_activity(hours=24)
        
        assert activity['total_predictions'] == 0
        assert activity['avg_confidence'] == 0.0
    
    def test_get_recent_activity_custom_hours(self, monitoring_service, mock_db_manager):
        """Test recent activity dengan custom hours"""
        mock_db_manager.execute_query.return_value = [
            {
                'total_predictions': 10,
                'avg_confidence': 0.80,
                'avg_latency': 0.2,
                'models_used': 2
            }
        ]
        
        activity = monitoring_service.get_recent_activity(hours=6)
        
        call_args = mock_db_manager.execute_query.call_args
        assert call_args[0][1] == (6,)
