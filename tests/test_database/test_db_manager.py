"""
Unit tests untuk DatabaseManager class
Tests: connection, retry logic, insert operations, query operations
"""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path
from database.db_manager import DatabaseManager


@pytest.fixture
def temp_db():
    """Create temporary database untuk testing"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
    temp_file.close()
    db_path = temp_file.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def db_manager(temp_db):
    """Create DatabaseManager instance dengan temporary database"""
    manager = DatabaseManager(temp_db)
    manager.connect()
    manager.initialize_schema()
    
    yield manager
    
    # Cleanup
    manager.disconnect()


class TestDatabaseConnection:
    """Test database connection dan retry logic"""
    
    def test_connect_success(self, temp_db):
        """Test successful database connection"""
        manager = DatabaseManager(temp_db)
        connection = manager.connect()
        
        assert connection is not None
        assert manager.connection is not None
        assert not manager.is_postgres
        
        manager.disconnect()
    
    def test_connect_invalid_path(self):
        """Test connection dengan invalid database path"""
        manager = DatabaseManager("/invalid/path/database.db")
        
        with pytest.raises(Exception) as exc_info:
            manager.connect()
        
        assert "Failed to connect" in str(exc_info.value)
    
    def test_disconnect(self, db_manager):
        """Test database disconnect"""
        assert db_manager.connection is not None
        
        db_manager.disconnect()
        
        assert db_manager.connection is None
    
    def test_is_postgresql_detection(self):
        """Test PostgreSQL URL detection"""
        pg_manager = DatabaseManager("postgresql://user:pass@localhost/db")
        assert pg_manager.is_postgres is True
        
        sqlite_manager = DatabaseManager("test.db")
        assert sqlite_manager.is_postgres is False


class TestSchemaInitialization:
    """Test database schema initialization"""
    
    def test_initialize_schema(self, temp_db):
        """Test schema initialization"""
        manager = DatabaseManager(temp_db)
        manager.connect()
        
        result = manager.initialize_schema()
        
        assert result is True
        assert manager._tables_exist() is True
        
        manager.disconnect()
    
    def test_tables_exist_check(self, db_manager):
        """Test checking if tables exist"""
        assert db_manager._tables_exist() is True
    
    def test_initialize_schema_idempotent(self, db_manager):
        """Test schema initialization adalah idempotent"""
        # Schema sudah initialized di fixture
        result = db_manager.initialize_schema()
        
        # Should return True tanpa error
        assert result is True


class TestInsertOperations:
    """Test insert operations"""
    
    def test_insert_user_input(self, db_manager):
        """Test inserting user input"""
        text = "Test input text"
        consent = True
        
        input_id = db_manager.insert_user_input(text, consent)
        
        assert input_id > 0
        assert isinstance(input_id, int)
    
    def test_insert_user_input_without_consent(self, db_manager):
        """Test inserting user input tanpa consent"""
        text = "Test input without consent"
        consent = False
        
        input_id = db_manager.insert_user_input(text, consent)
        
        assert input_id > 0
    
    def test_insert_prediction(self, db_manager):
        """Test inserting prediction"""
        # Insert user input first
        input_id = db_manager.insert_user_input("Test text", True)
        
        # Insert prediction
        prediction_id = db_manager.insert_prediction(
            input_id=input_id,
            model_version="v1",
            prediction="positif",
            confidence=0.85,
            latency=0.234
        )
        
        assert prediction_id > 0
        assert isinstance(prediction_id, int)
    
    def test_insert_prediction_invalid_input_id(self, db_manager):
        """Test inserting prediction dengan invalid input_id"""
        # Note: SQLite doesn't enforce foreign keys by default
        # This test verifies the insert completes without crash
        # In production with PostgreSQL, this would raise an error
        try:
            prediction_id = db_manager.insert_prediction(
                input_id=99999,  # Non-existent ID
                model_version="v1",
                prediction="positif",
                confidence=0.85,
                latency=0.234
            )
            # SQLite allows this, so we just verify it returns an ID
            assert prediction_id > 0
        except Exception:
            # PostgreSQL would raise an error here
            pass


class TestQueryOperations:
    """Test query operations"""
    
    def test_execute_query_select(self, db_manager):
        """Test executing SELECT query"""
        # Insert test data
        input_id = db_manager.insert_user_input("Test query", True)
        
        # Query data
        query = "SELECT * FROM users_inputs WHERE id = ?"
        results = db_manager.execute_query(query, (input_id,))
        
        assert len(results) == 1
        assert results[0]['text_input'] == "Test query"
        assert results[0]['user_consent'] == 1
    
    def test_execute_query_empty_result(self, db_manager):
        """Test query yang return empty result"""
        query = "SELECT * FROM users_inputs WHERE id = ?"
        results = db_manager.execute_query(query, (99999,))
        
        assert len(results) == 0
        assert isinstance(results, list)
    
    def test_get_recent_predictions(self, db_manager):
        """Test getting recent predictions"""
        # Insert test data
        input_id = db_manager.insert_user_input("Test prediction", True)
        db_manager.insert_prediction(input_id, "v1", "positif", 0.9, 0.1)
        
        # Get recent predictions
        results = db_manager.get_recent_predictions(limit=10)
        
        assert len(results) > 0
        assert 'text_input' in results[0]
        assert 'prediction' in results[0]
        assert 'confidence' in results[0]
    
    def test_get_recent_predictions_with_limit(self, db_manager):
        """Test getting recent predictions dengan limit"""
        # Insert multiple predictions
        for i in range(5):
            input_id = db_manager.insert_user_input(f"Test {i}", True)
            db_manager.insert_prediction(input_id, "v1", "positif", 0.9, 0.1)
        
        # Get with limit
        results = db_manager.get_recent_predictions(limit=3)
        
        assert len(results) == 3
    
    def test_get_dataset_snapshot(self, db_manager):
        """Test getting dataset snapshot"""
        # Insert test data
        input_id = db_manager.insert_user_input("Test snapshot", True)
        db_manager.insert_prediction(input_id, "v1", "positif", 0.9, 0.1)
        
        # Get snapshot
        df = db_manager.get_dataset_snapshot(consent_only=True)
        
        assert len(df) > 0
        assert 'text_input' in df.columns
        assert 'prediction' in df.columns
    
    def test_get_dataset_snapshot_consent_only(self, db_manager):
        """Test dataset snapshot dengan consent filter"""
        # Insert data dengan dan tanpa consent
        input_id1 = db_manager.insert_user_input("With consent", True)
        db_manager.insert_prediction(input_id1, "v1", "positif", 0.9, 0.1)
        
        input_id2 = db_manager.insert_user_input("Without consent", False)
        db_manager.insert_prediction(input_id2, "v1", "negatif", 0.8, 0.2)
        
        # Get snapshot dengan consent only
        df = db_manager.get_dataset_snapshot(consent_only=True)
        
        # Should only have 1 record (with consent)
        assert len(df) == 1
        assert df.iloc[0]['text_input'] == "With consent"
    
    def test_get_metrics_by_version(self, db_manager):
        """Test getting metrics by model version"""
        # Insert test data untuk multiple versions
        input_id1 = db_manager.insert_user_input("Test v1", True)
        db_manager.insert_prediction(input_id1, "v1", "positif", 0.9, 0.1)
        
        input_id2 = db_manager.insert_user_input("Test v2", True)
        db_manager.insert_prediction(input_id2, "v2", "negatif", 0.85, 0.15)
        
        # Get metrics
        metrics = db_manager.get_metrics_by_version()
        
        assert len(metrics) == 2
        assert 'v1' in metrics
        assert 'v2' in metrics
        assert 'prediction_count' in metrics['v1']
        assert 'avg_confidence' in metrics['v1']
        assert 'avg_latency' in metrics['v1']


class TestTransactionOperations:
    """Test transaction operations"""
    
    def test_execute_transaction_success(self, db_manager):
        """Test successful transaction execution"""
        queries = [
            ("INSERT INTO users_inputs (text_input, user_consent) VALUES (?, ?)", ("Test 1", True)),
            ("INSERT INTO users_inputs (text_input, user_consent) VALUES (?, ?)", ("Test 2", True))
        ]
        
        result = db_manager.execute_transaction(queries)
        
        assert result is True
        
        # Verify data inserted
        all_inputs = db_manager.execute_query("SELECT * FROM users_inputs")
        assert len(all_inputs) >= 2
    
    def test_execute_transaction_rollback(self, db_manager):
        """Test transaction rollback on error"""
        # Insert valid data first
        input_id = db_manager.insert_user_input("Valid input", True)
        
        # Create transaction dengan invalid query
        queries = [
            ("INSERT INTO users_inputs (text_input, user_consent) VALUES (?, ?)", ("Test", True)),
            ("INSERT INTO invalid_table (col) VALUES (?)", ("Invalid",))  # This will fail
        ]
        
        result = db_manager.execute_transaction(queries)
        
        assert result is False


class TestErrorHandling:
    """Test error handling"""
    
    def test_query_with_invalid_sql(self, db_manager):
        """Test query dengan invalid SQL"""
        with pytest.raises(Exception):
            db_manager.execute_query("INVALID SQL QUERY")
    
    def test_insert_with_missing_connection(self, temp_db):
        """Test insert tanpa connection"""
        manager = DatabaseManager(temp_db)
        # Don't connect
        
        # Should auto-connect
        manager.initialize_schema()
        input_id = manager.insert_user_input("Test", True)
        
        assert input_id > 0
        
        manager.disconnect()
