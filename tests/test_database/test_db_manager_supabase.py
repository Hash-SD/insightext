"""
Integration tests untuk DatabaseManager dengan Supabase PostgreSQL
Tests: connection, schema, insert, query operations dengan real Supabase database

IMPORTANT: These tests connect to real Supabase database
- Requires valid DATABASE_URL in .env file
- Tests will create and cleanup test data
- Use with caution in production environment
"""

import pytest
import os
from datetime import datetime
from dotenv import load_dotenv
from database.db_manager import DatabaseManager

# Load environment variables
# Override=False means .env file takes precedence over existing env vars
load_dotenv(override=True)

# Skip tests jika DATABASE_URL tidak tersedia atau bukan PostgreSQL
DATABASE_URL = os.getenv('DATABASE_URL', '')

# Check if it's a valid Supabase/PostgreSQL URL (not localhost)
is_valid_supabase = (
    (DATABASE_URL.startswith('postgresql://') or DATABASE_URL.startswith('postgres://')) and
    'localhost' not in DATABASE_URL and
    'supabase' in DATABASE_URL
)

SKIP_SUPABASE_TESTS = not is_valid_supabase

skip_reason = "Supabase DATABASE_URL not configured properly."
if not DATABASE_URL:
    skip_reason += " DATABASE_URL is not set in .env file."
elif 'localhost' in DATABASE_URL:
    skip_reason += " DATABASE_URL points to localhost, not Supabase."
elif 'supabase' not in DATABASE_URL:
    skip_reason += " DATABASE_URL is not a Supabase URL."
else:
    skip_reason += f" Current URL: {DATABASE_URL[:50]}..."

skip_reason += " See tests/SUPABASE_SETUP.md for setup instructions."

pytestmark = pytest.mark.skipif(
    SKIP_SUPABASE_TESTS,
    reason=skip_reason
)


@pytest.fixture(scope="module")
def supabase_db_manager():
    """
    Create DatabaseManager instance dengan Supabase connection.
    Scope: module - reuse connection untuk semua tests dalam module ini.
    """
    if SKIP_SUPABASE_TESTS:
        pytest.skip("Supabase tests skipped")
    
    manager = DatabaseManager(DATABASE_URL)
    manager.connect()
    
    # Ensure schema exists
    manager.initialize_schema()
    
    yield manager
    
    # Cleanup: disconnect
    manager.disconnect()


@pytest.fixture
def cleanup_test_data(supabase_db_manager):
    """
    Fixture untuk cleanup test data setelah each test.
    Menggunakan timestamp untuk identify test data.
    """
    test_start_time = datetime.now()
    
    yield
    
    # Cleanup: delete test data created during this test
    # Delete predictions first (foreign key constraint)
    try:
        cleanup_query = """
            DELETE FROM predictions 
            WHERE input_id IN (
                SELECT id FROM users_inputs 
                WHERE timestamp >= %s
            )
        """
        supabase_db_manager.execute_query(cleanup_query, (test_start_time,))
        
        cleanup_query = "DELETE FROM users_inputs WHERE timestamp >= %s"
        supabase_db_manager.execute_query(cleanup_query, (test_start_time,))
        
        supabase_db_manager.connection.commit()
    except Exception as e:
        print(f"Cleanup warning: {e}")


class TestSupabaseConnection:
    """Test Supabase PostgreSQL connection"""
    
    def test_connect_to_supabase(self, supabase_db_manager):
        """Test successful connection ke Supabase"""
        assert supabase_db_manager.connection is not None
        assert supabase_db_manager.is_postgres is True
        assert supabase_db_manager.db_url.startswith('postgresql://') or \
               supabase_db_manager.db_url.startswith('postgres://')
    
    def test_connection_properties(self, supabase_db_manager):
        """Test connection properties"""
        # Verify connection is active
        cursor = supabase_db_manager.connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        
        assert result is not None
    
    def test_database_info(self, supabase_db_manager):
        """Test retrieve database information"""
        query = "SELECT version()"
        results = supabase_db_manager.execute_query(query)
        
        assert len(results) > 0
        assert 'PostgreSQL' in results[0]['version']


class TestSupabaseSchema:
    """Test schema operations di Supabase"""
    
    def test_tables_exist(self, supabase_db_manager):
        """Test bahwa required tables exist"""
        assert supabase_db_manager._tables_exist() is True
    
    def test_users_inputs_table_structure(self, supabase_db_manager):
        """Test struktur tabel users_inputs"""
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'users_inputs'
            ORDER BY ordinal_position
        """
        columns = supabase_db_manager.execute_query(query)
        
        assert len(columns) > 0
        
        # Verify required columns exist
        column_names = [col['column_name'] for col in columns]
        assert 'id' in column_names
        assert 'timestamp' in column_names
        assert 'text_input' in column_names
        assert 'user_consent' in column_names
        assert 'anonymized' in column_names
    
    def test_predictions_table_structure(self, supabase_db_manager):
        """Test struktur tabel predictions"""
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'predictions'
            ORDER BY ordinal_position
        """
        columns = supabase_db_manager.execute_query(query)
        
        assert len(columns) > 0
        
        # Verify required columns exist
        column_names = [col['column_name'] for col in columns]
        assert 'id' in column_names
        assert 'input_id' in column_names
        assert 'model_version' in column_names
        assert 'prediction' in column_names
        assert 'confidence' in column_names
        assert 'latency' in column_names
        assert 'timestamp' in column_names
    
    def test_foreign_key_constraint(self, supabase_db_manager):
        """Test foreign key constraint antara predictions dan users_inputs"""
        query = """
            SELECT
                tc.constraint_name,
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = 'predictions'
        """
        constraints = supabase_db_manager.execute_query(query)
        
        assert len(constraints) > 0
        assert any(c['foreign_table_name'] == 'users_inputs' for c in constraints)
    
    def test_indexes_exist(self, supabase_db_manager):
        """Test bahwa indexes exist untuk performance"""
        query = """
            SELECT indexname, tablename
            FROM pg_indexes
            WHERE tablename IN ('users_inputs', 'predictions')
        """
        indexes = supabase_db_manager.execute_query(query)
        
        assert len(indexes) > 0


class TestSupabaseInsertOperations:
    """Test insert operations ke Supabase"""
    
    def test_insert_user_input(self, supabase_db_manager, cleanup_test_data):
        """Test inserting user input ke Supabase"""
        text = "Test input untuk Supabase"
        consent = True
        
        input_id = supabase_db_manager.insert_user_input(text, consent)
        
        assert input_id > 0
        assert isinstance(input_id, int)
        
        # Verify data inserted
        query = "SELECT * FROM users_inputs WHERE id = %s"
        results = supabase_db_manager.execute_query(query, (input_id,))
        
        assert len(results) == 1
        assert results[0]['text_input'] == text
        assert results[0]['user_consent'] is True
    
    def test_insert_user_input_without_consent(self, supabase_db_manager, cleanup_test_data):
        """Test inserting user input tanpa consent"""
        text = "Test tanpa consent"
        consent = False
        
        input_id = supabase_db_manager.insert_user_input(text, consent)
        
        assert input_id > 0
        
        # Verify consent flag
        query = "SELECT user_consent FROM users_inputs WHERE id = %s"
        results = supabase_db_manager.execute_query(query, (input_id,))
        
        assert results[0]['user_consent'] is False
    
    def test_insert_prediction(self, supabase_db_manager, cleanup_test_data):
        """Test inserting prediction ke Supabase"""
        # Insert user input first
        input_id = supabase_db_manager.insert_user_input("Test prediction", True)
        
        # Insert prediction
        prediction_id = supabase_db_manager.insert_prediction(
            input_id=input_id,
            model_version="v1",
            prediction="positif",
            confidence=0.85,
            latency=0.234
        )
        
        assert prediction_id > 0
        
        # Verify data inserted
        query = "SELECT * FROM predictions WHERE id = %s"
        results = supabase_db_manager.execute_query(query, (prediction_id,))
        
        assert len(results) == 1
        assert results[0]['model_version'] == "v1"
        assert results[0]['prediction'] == "positif"
        assert abs(results[0]['confidence'] - 0.85) < 0.001
    
    def test_insert_prediction_foreign_key_violation(self, supabase_db_manager):
        """Test bahwa foreign key constraint enforced di PostgreSQL"""
        # Try to insert prediction dengan non-existent input_id
        with pytest.raises(Exception) as exc_info:
            supabase_db_manager.insert_prediction(
                input_id=999999,  # Non-existent ID
                model_version="v1",
                prediction="positif",
                confidence=0.85,
                latency=0.234
            )
        
        # PostgreSQL should raise foreign key violation error
        assert "foreign key" in str(exc_info.value).lower() or \
               "violates" in str(exc_info.value).lower()
    
    def test_insert_multiple_predictions(self, supabase_db_manager, cleanup_test_data):
        """Test inserting multiple predictions"""
        # Insert user input
        input_id = supabase_db_manager.insert_user_input("Test multiple", True)
        
        # Insert multiple predictions
        prediction_ids = []
        for i in range(3):
            pred_id = supabase_db_manager.insert_prediction(
                input_id=input_id,
                model_version=f"v{i+1}",
                prediction="positif",
                confidence=0.8 + (i * 0.05),
                latency=0.2 + (i * 0.1)
            )
            prediction_ids.append(pred_id)
        
        assert len(prediction_ids) == 3
        assert all(pid > 0 for pid in prediction_ids)


class TestSupabaseQueryOperations:
    """Test query operations dari Supabase"""
    
    def test_get_recent_predictions(self, supabase_db_manager, cleanup_test_data):
        """Test retrieving recent predictions"""
        # Insert test data
        input_id = supabase_db_manager.insert_user_input("Test query", True)
        supabase_db_manager.insert_prediction(input_id, "v1", "positif", 0.9, 0.1)
        
        # Get recent predictions
        results = supabase_db_manager.get_recent_predictions(limit=10)
        
        assert len(results) > 0
        assert 'text_input' in results[0]
        assert 'prediction' in results[0]
        assert 'model_version' in results[0]
    
    def test_get_recent_predictions_with_limit(self, supabase_db_manager, cleanup_test_data):
        """Test limit parameter"""
        # Insert multiple predictions
        for i in range(5):
            input_id = supabase_db_manager.insert_user_input(f"Test {i}", True)
            supabase_db_manager.insert_prediction(input_id, "v1", "positif", 0.9, 0.1)
        
        # Get with limit
        results = supabase_db_manager.get_recent_predictions(limit=3)
        
        assert len(results) <= 3
    
    def test_get_dataset_snapshot(self, supabase_db_manager, cleanup_test_data):
        """Test getting dataset snapshot"""
        # Insert test data dengan consent
        input_id = supabase_db_manager.insert_user_input("Test snapshot", True)
        supabase_db_manager.insert_prediction(input_id, "v1", "positif", 0.9, 0.1)
        
        # Get snapshot
        df = supabase_db_manager.get_dataset_snapshot(consent_only=True)
        
        assert len(df) > 0
        assert 'text_input' in df.columns
        assert 'prediction' in df.columns
    
    def test_get_dataset_snapshot_consent_filter(self, supabase_db_manager, cleanup_test_data):
        """Test consent filter dalam dataset snapshot"""
        # Insert data dengan consent
        input_id1 = supabase_db_manager.insert_user_input("With consent", True)
        supabase_db_manager.insert_prediction(input_id1, "v1", "positif", 0.9, 0.1)
        
        # Insert data tanpa consent
        input_id2 = supabase_db_manager.insert_user_input("Without consent", False)
        supabase_db_manager.insert_prediction(input_id2, "v1", "negatif", 0.8, 0.2)
        
        # Get snapshot dengan consent only
        df = supabase_db_manager.get_dataset_snapshot(consent_only=True)
        
        # Verify only consent data included
        consent_texts = df['text_input'].tolist()
        assert "With consent" in consent_texts
        assert "Without consent" not in consent_texts
    
    def test_get_metrics_by_version(self, supabase_db_manager, cleanup_test_data):
        """Test getting aggregated metrics"""
        # Insert test data untuk multiple versions
        input_id1 = supabase_db_manager.insert_user_input("Test v1", True)
        supabase_db_manager.insert_prediction(input_id1, "v1", "positif", 0.9, 0.1)
        
        input_id2 = supabase_db_manager.insert_user_input("Test v2", True)
        supabase_db_manager.insert_prediction(input_id2, "v2", "negatif", 0.85, 0.15)
        
        # Get metrics
        metrics = supabase_db_manager.get_metrics_by_version()
        
        assert len(metrics) >= 2
        assert 'v1' in metrics
        assert 'v2' in metrics
        assert 'prediction_count' in metrics['v1']
        assert 'avg_confidence' in metrics['v1']


class TestSupabaseTransactions:
    """Test transaction operations di Supabase"""
    
    def test_transaction_commit(self, supabase_db_manager, cleanup_test_data):
        """Test successful transaction commit"""
        queries = [
            ("INSERT INTO users_inputs (text_input, user_consent) VALUES (%s, %s)", ("Test 1", True)),
            ("INSERT INTO users_inputs (text_input, user_consent) VALUES (%s, %s)", ("Test 2", True))
        ]
        
        result = supabase_db_manager.execute_transaction(queries)
        
        assert result is True
    
    def test_transaction_rollback(self, supabase_db_manager):
        """Test transaction rollback on error"""
        # Create transaction dengan invalid query
        queries = [
            ("INSERT INTO users_inputs (text_input, user_consent) VALUES (%s, %s)", ("Test", True)),
            ("INSERT INTO invalid_table (col) VALUES (%s)", ("Invalid",))  # This will fail
        ]
        
        result = supabase_db_manager.execute_transaction(queries)
        
        assert result is False


class TestSupabasePerformance:
    """Test performance characteristics di Supabase"""
    
    def test_bulk_insert_performance(self, supabase_db_manager, cleanup_test_data):
        """Test performance untuk bulk inserts"""
        import time
        
        start_time = time.time()
        
        # Insert 10 records
        for i in range(10):
            input_id = supabase_db_manager.insert_user_input(f"Bulk test {i}", True)
            supabase_db_manager.insert_prediction(input_id, "v1", "positif", 0.9, 0.1)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds for 10 inserts)
        assert elapsed_time < 5.0
        
        print(f"\nBulk insert performance: {elapsed_time:.2f}s for 10 records")
    
    def test_query_performance(self, supabase_db_manager):
        """Test query performance"""
        import time
        
        start_time = time.time()
        
        # Execute query
        results = supabase_db_manager.get_recent_predictions(limit=100)
        
        elapsed_time = time.time() - start_time
        
        # Should complete quickly (< 1 second)
        assert elapsed_time < 1.0
        
        print(f"\nQuery performance: {elapsed_time:.3f}s for recent predictions")


class TestSupabaseConnectionResilience:
    """Test connection resilience dan error handling"""
    
    def test_reconnect_after_disconnect(self, supabase_db_manager):
        """Test reconnection setelah disconnect"""
        # Disconnect
        supabase_db_manager.disconnect()
        assert supabase_db_manager.connection is None
        
        # Reconnect
        supabase_db_manager.connect()
        assert supabase_db_manager.connection is not None
        
        # Verify connection works
        query = "SELECT 1"
        results = supabase_db_manager.execute_query(query)
        assert len(results) > 0
    
    def test_query_auto_reconnect(self, supabase_db_manager):
        """Test auto-reconnect saat query jika connection lost"""
        # Disconnect
        supabase_db_manager.disconnect()
        
        # Query should auto-reconnect
        results = supabase_db_manager.get_recent_predictions(limit=1)
        
        # Should work (auto-reconnected)
        assert isinstance(results, list)


class TestSupabaseDataIntegrity:
    """Test data integrity di Supabase"""
    
    def test_timestamp_auto_generation(self, supabase_db_manager, cleanup_test_data):
        """Test bahwa timestamp auto-generated"""
        input_id = supabase_db_manager.insert_user_input("Test timestamp", True)
        
        query = "SELECT timestamp FROM users_inputs WHERE id = %s"
        results = supabase_db_manager.execute_query(query, (input_id,))
        
        assert results[0]['timestamp'] is not None
    
    def test_data_type_validation(self, supabase_db_manager, cleanup_test_data):
        """Test data type validation"""
        input_id = supabase_db_manager.insert_user_input("Test types", True)
        
        # Insert prediction dengan correct types
        pred_id = supabase_db_manager.insert_prediction(
            input_id=input_id,
            model_version="v1",
            prediction="positif",
            confidence=0.85,
            latency=0.234
        )
        
        # Verify types preserved
        query = "SELECT confidence, latency FROM predictions WHERE id = %s"
        results = supabase_db_manager.execute_query(query, (pred_id,))
        
        assert isinstance(results[0]['confidence'], (float, int))
        assert isinstance(results[0]['latency'], (float, int))
    
    def test_text_encoding_utf8(self, supabase_db_manager, cleanup_test_data):
        """Test UTF-8 encoding untuk Bahasa Indonesia"""
        text_with_special_chars = "Ini teks dengan karakter spesial: é, ñ, ü, 你好, 🎉"
        
        input_id = supabase_db_manager.insert_user_input(text_with_special_chars, True)
        
        # Retrieve and verify
        query = "SELECT text_input FROM users_inputs WHERE id = %s"
        results = supabase_db_manager.execute_query(query, (input_id,))
        
        assert results[0]['text_input'] == text_with_special_chars


# Summary test untuk verify overall Supabase integration
class TestSupabaseIntegrationSummary:
    """Summary test untuk verify Supabase integration"""
    
    def test_complete_workflow(self, supabase_db_manager, cleanup_test_data):
        """Test complete workflow: insert → query → aggregate"""
        # Step 1: Insert user input
        input_id = supabase_db_manager.insert_user_input(
            "Complete workflow test", 
            True
        )
        assert input_id > 0
        
        # Step 2: Insert prediction
        pred_id = supabase_db_manager.insert_prediction(
            input_id=input_id,
            model_version="v1",
            prediction="positif",
            confidence=0.92,
            latency=0.156
        )
        assert pred_id > 0
        
        # Step 3: Query recent predictions
        recent = supabase_db_manager.get_recent_predictions(limit=1)
        assert len(recent) > 0
        
        # Step 4: Get dataset snapshot
        df = supabase_db_manager.get_dataset_snapshot(consent_only=True)
        assert len(df) > 0
        
        # Step 5: Get metrics
        metrics = supabase_db_manager.get_metrics_by_version()
        assert 'v1' in metrics
        
        print("\n✓ Complete Supabase workflow test passed!")
