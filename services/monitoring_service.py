"""Monitoring service for aggregating metrics and monitoring model performance."""

import logging
import statistics
from typing import Dict, List, Any, Optional

from database.db_manager import DatabaseManager


class MonitoringService:
    """Service for monitoring model performance and aggregating metrics."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated metrics per model version."""
        try:
            self.logger.info("Retrieving metrics summary")
            metrics = self.db_manager.get_metrics_by_version()
            
            if not metrics:
                self.logger.warning("No metrics found in database")
                return {}
            
            self.logger.info(f"Retrieved metrics for {len(metrics)} model versions")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error retrieving metrics summary: {e}", exc_info=True)
            return {}
    
    def get_latency_distribution(self, model_version: Optional[str] = None) -> List[float]:
        """Get latency distribution for histogram data."""
        try:
            self.logger.info(f"Retrieving latency distribution for model: {model_version or 'all'}")
            
            if model_version:
                query = "SELECT latency FROM predictions WHERE model_version = ? ORDER BY timestamp DESC"
                params = (model_version,)
            else:
                query = "SELECT latency FROM predictions ORDER BY timestamp DESC"
                params = ()
            
            results = self.db_manager.execute_query(query, params)
            latencies = [row['latency'] for row in results]
            
            self.logger.info(f"Retrieved {len(latencies)} latency values")
            return latencies
            
        except Exception as e:
            self.logger.error(f"Error retrieving latency distribution: {e}", exc_info=True)
            return []
    
    def calculate_drift_score(self) -> float:
        """
        Calculate data drift score based on recent prediction statistics.
        Uses confidence score variance as a proxy for drift.
        """
        try:
            self.logger.info("Calculating drift score from database statistics")
            
            recent_query = "SELECT confidence, prediction FROM predictions ORDER BY timestamp DESC LIMIT 50"
            baseline_query = "SELECT confidence, prediction FROM predictions ORDER BY timestamp DESC LIMIT 500"
            
            recent_data = self.db_manager.execute_query(recent_query)
            baseline_data = self.db_manager.execute_query(baseline_query)
            
            if len(recent_data) < 10 or len(baseline_data) < 50:
                self.logger.warning("Insufficient data for drift calculation")
                return 0.1
            
            recent_confidences = [row['confidence'] for row in recent_data]
            baseline_confidences = [row['confidence'] for row in baseline_data]
            
            recent_mean = statistics.mean(recent_confidences)
            baseline_mean = statistics.mean(baseline_confidences)
            recent_stdev = statistics.stdev(recent_confidences) if len(recent_confidences) > 1 else 0
            baseline_stdev = statistics.stdev(baseline_confidences) if len(baseline_confidences) > 1 else 0
            
            # Calculate drift components
            mean_shift = abs(recent_mean - baseline_mean)
            variance_change = abs(recent_stdev - baseline_stdev) / (baseline_stdev + 0.01)
            
            # Combine into drift score
            drift_score = min(0.6 * mean_shift + 0.4 * variance_change, 1.0)
            
            self.logger.info(f"Drift score: {drift_score:.4f} (mean_shift={mean_shift:.4f}, var_change={variance_change:.4f})")
            return drift_score
            
        except Exception as e:
            self.logger.error(f"Error calculating drift score: {e}", exc_info=True)
            return 0.0
    
    def get_prediction_counts(self) -> Dict[str, int]:
        """Get prediction counts per model version."""
        try:
            query = "SELECT model_version, COUNT(*) as count FROM predictions GROUP BY model_version ORDER BY model_version"
            results = self.db_manager.execute_query(query)
            counts = {row['model_version']: row['count'] for row in results}
            
            self.logger.info(f"Retrieved prediction counts for {len(counts)} versions")
            return counts
            
        except Exception as e:
            self.logger.error(f"Error retrieving prediction counts: {e}", exc_info=True)
            return {}
    
    def get_confidence_distribution(self, model_version: Optional[str] = None) -> List[float]:
        """Get confidence score distribution."""
        try:
            if model_version:
                query = "SELECT confidence FROM predictions WHERE model_version = ? ORDER BY timestamp DESC"
                params = (model_version,)
            else:
                query = "SELECT confidence FROM predictions ORDER BY timestamp DESC"
                params = ()
            
            results = self.db_manager.execute_query(query, params)
            return [row['confidence'] for row in results]
            
        except Exception as e:
            self.logger.error(f"Error retrieving confidence distribution: {e}", exc_info=True)
            return []
    
    def get_prediction_timeline(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get prediction timeline for time-series visualization."""
        try:
            query = """
                SELECT timestamp, model_version, confidence, latency
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            """
            return self.db_manager.execute_query(query, (limit,))
            
        except Exception as e:
            self.logger.error(f"Error retrieving prediction timeline: {e}", exc_info=True)
            return []
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get comprehensive comparison of all model versions."""
        try:
            metrics_summary = self.get_metrics_summary()
            prediction_counts = self.get_prediction_counts()
            
            comparison = {}
            for version in metrics_summary.keys():
                comparison[version] = {
                    **metrics_summary[version],
                    'total_predictions': prediction_counts.get(version, 0)
                }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error generating model comparison: {e}", exc_info=True)
            return {}
    
    def get_recent_activity(self, hours: int = 24) -> Dict[str, Any]:
        """Get recent activity statistics."""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(confidence) as avg_confidence,
                    AVG(latency) as avg_latency,
                    COUNT(DISTINCT model_version) as models_used
                FROM predictions
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """
            results = self.db_manager.execute_query(query, (hours,))
            
            if results:
                return dict(results[0])
            
            return {
                'total_predictions': 0,
                'avg_confidence': 0.0,
                'avg_latency': 0.0,
                'models_used': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent activity: {e}", exc_info=True)
            return {
                'total_predictions': 0,
                'avg_confidence': 0.0,
                'avg_latency': 0.0,
                'models_used': 0
            }
    
    def get_dashboard_data(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Batch fetch all dashboard data in a single method call.
        Reduces multiple database round-trips to one batched operation.
        """
        try:
            self.logger.info("Batch fetching dashboard data (optimized)")
            
            metrics_summary = self.db_manager.get_metrics_by_version() or {}
            
            if model_version:
                latency_query = "SELECT latency FROM predictions WHERE model_version = ? ORDER BY timestamp DESC"
                latency_results = self.db_manager.execute_query(latency_query, (model_version,))
            else:
                latency_query = "SELECT latency FROM predictions ORDER BY timestamp DESC"
                latency_results = self.db_manager.execute_query(latency_query)
            
            latency_data = [row['latency'] for row in latency_results]
            drift_score = self.calculate_drift_score()
            
            self.logger.info(f"Dashboard data fetched: {len(metrics_summary)} versions, {len(latency_data)} latency records")
            
            return {
                'metrics_summary': metrics_summary,
                'latency_data': latency_data,
                'drift_score': drift_score
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching dashboard data: {e}", exc_info=True)
            return {
                'metrics_summary': {},
                'latency_data': [],
                'drift_score': 0.0
            }
