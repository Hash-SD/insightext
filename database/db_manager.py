"""
Database Manager dengan support untuk PostgreSQL (Supabase) dan SQLite.
Automatically detects database type dari connection string.
"""

import time
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manager untuk database operations dengan retry logic dan transaction support"""
    
    def __init__(self, db_url: str = "mlops_app.db"):
        """
        Initialize database manager
        
        Args:
            db_url: Path ke database file (SQLite) atau connection string (PostgreSQL)
        """
        self.db_url = db_url
        self.connection: Optional[Any] = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.is_postgres = self._is_postgresql()
        
        # Import appropriate library
        if self.is_postgres:
            import psycopg2
            import psycopg2.extras
            self.db_module = psycopg2
            self.extras = psycopg2.extras
        else:
            import sqlite3
            self.db_module = sqlite3
            self.extras = None
    
    def _is_postgresql(self) -> bool:
        """Check if database URL is PostgreSQL"""
        return self.db_url.startswith('postgresql://') or self.db_url.startswith('postgres://')
    
    def connect(self) -> Any:
        """
        Establish database connection dengan retry logic
        
        Returns:
            Connection object (psycopg2 atau sqlite3)
            
        Raises:
            Exception: Jika connection gagal setelah max retries
        """
        retries = 0
        last_error = None
        attempted_pooled = False
        
        while retries < self.max_retries:
            try:
                if self.is_postgres:
                    # Primary attempt: whatever db_url provided (direct or pooled)
                    url = self.db_url
                    # Append sslmode if missing
                    if 'sslmode=' not in url:
                        sep = '?' if '?' not in url else '&'
                        url = f"{url}{sep}sslmode=require"
                    try:
                        self.connection = self.db_module.connect(
                            url,
                            cursor_factory=self.extras.RealDictCursor
                        )
                        self.connection.autocommit = False
                        logger.info(
                            f"PostgreSQL connection established: {url.split('@')[1] if '@' in url else 'database'}"
                        )
                    except Exception as primary_error:
                        # Fallback: build pooled URL if original looked like direct and not tried yet
                        if (not attempted_pooled and 'pooler.supabase.com' not in self.db_url
                                and 'db.' in self.db_url and '.supabase.co' in self.db_url):
                            attempted_pooled = True
                            import re
                            host_match = re.search(r"db\.([a-z0-9]+)\.supabase\.co", self.db_url)
                            pwd_match = re.search(r"postgresql://[^:]+:([^@]+)@", self.db_url)
                            project_ref = host_match.group(1) if host_match else None
                            password = pwd_match.group(1) if pwd_match else None
                            if project_ref and password:
                                pooled_url = f"postgresql://postgres.{project_ref}:{password}@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
                                if 'sslmode=' not in pooled_url:
                                    pooled_url += '?sslmode=require'
                                logger.warning(
                                    f"Primary PostgreSQL connect failed ({primary_error}); trying pooled URL: {pooled_url}"\
                                )
                                self.connection = self.db_module.connect(
                                    pooled_url,
                                    cursor_factory=self.extras.RealDictCursor
                                )
                                self.connection.autocommit = False
                                logger.info(
                                    f"PostgreSQL pooled connection established: {pooled_url.split('@')[1] if '@' in pooled_url else 'database'}"
                                )
                            else:
                                raise primary_error
                        else:
                            raise primary_error
                else:
                    # SQLite connection
                    self.connection = self.db_module.connect(self.db_url, check_same_thread=False)
                    self.connection.row_factory = self.db_module.Row
                    logger.info(f"SQLite connection established: {self.db_url}")
                
                return self.connection
                
            except Exception as e:
                retries += 1
                last_error = e
                wait_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                logger.warning(
                    f"Database connection attempt {retries}/{self.max_retries} failed: {e}. "
                    f"Retrying in {wait_time} seconds..."
                )
                if retries < self.max_retries:
                    time.sleep(wait_time)
        
        error_msg = f"Failed to connect to database after {self.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
                self.connection = None
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
    
    def _convert_query_params(self, query: str, params: tuple) -> Tuple[str, tuple]:
        """
        Convert query parameters dari ? (SQLite) ke %s (PostgreSQL) jika perlu
        
        Args:
            query: SQL query dengan ? placeholders
            params: Query parameters
            
        Returns:
            Tuple[str, tuple]: Converted query dan params
        """
        if self.is_postgres and '?' in query:
            # Replace ? with %s for PostgreSQL
            query = query.replace('?', '%s')
        return query, params
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """
        Execute SELECT query dan return results
        
        Args:
            query: SQL SELECT query
            params: Query parameters untuk parameterized queries
            
        Returns:
            List[Dict]: List of dictionaries dengan query results
        """
        try:
            if not self.connection:
                self.connect()
            
            # Convert query parameters if needed
            query, params = self._convert_query_params(query, params)
            
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            if self.is_postgres:
                results = [dict(row) for row in rows]
            else:
                results = [dict(row) for row in rows]
            
            cursor.close()
            
            logger.debug(f"Query executed successfully: {query[:50]}... returned {len(results)} rows")
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {e}\nQuery: {query}\nParams: {params}")
            raise
    
    def execute_transaction(self, queries: List[Tuple[str, tuple]]) -> bool:
        """
        Execute multiple queries dalam single transaction
        
        Args:
            queries: List of tuples (query, params)
            
        Returns:
            bool: True jika transaction berhasil, False jika gagal
        """
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            
            # Start transaction
            for query, params in queries:
                query, params = self._convert_query_params(query, params)
                cursor.execute(query, params)
            
            # Commit transaction
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Transaction completed successfully: {len(queries)} queries executed")
            return True
            
        except Exception as e:
            # Rollback on error
            if self.connection:
                self.connection.rollback()
            logger.error(f"Transaction failed, rolled back: {e}")
            return False
    
    def insert_user_input(self, text: str, consent: bool) -> int:
        """
        Insert user input ke database
        
        Args:
            text: Input teks dari pengguna
            consent: User consent untuk menyimpan data
            
        Returns:
            int: ID dari inserted record
            
        Raises:
            Exception: Jika insert gagal
        """
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            
            if self.is_postgres:
                query = """
                    INSERT INTO users_inputs (text_input, user_consent, anonymized)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """
                cursor.execute(query, (text, consent, False))
                input_id = cursor.fetchone()['id']
            else:
                query = """
                    INSERT INTO users_inputs (text_input, user_consent, anonymized)
                    VALUES (?, ?, ?)
                """
                cursor.execute(query, (text, consent, False))
                input_id = cursor.lastrowid
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"User input inserted successfully: ID={input_id}, consent={consent}")
            return input_id
            
        except Exception as e:
            logger.error(f"Error inserting user input: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def insert_prediction(
        self, 
        input_id: int, 
        model_version: str, 
        prediction: str, 
        confidence: float, 
        latency: float
    ) -> int:
        """
        Insert prediction result ke database
        
        Args:
            input_id: ID dari user input yang terkait
            model_version: Versi model yang digunakan
            prediction: Hasil prediksi
            confidence: Confidence score (0-1)
            latency: Waktu prediksi dalam seconds
            
        Returns:
            int: ID dari inserted prediction record
            
        Raises:
            Exception: Jika insert gagal
        """
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            
            if self.is_postgres:
                query = """
                    INSERT INTO predictions (input_id, model_version, prediction, confidence, latency)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """
                cursor.execute(query, (input_id, model_version, prediction, confidence, latency))
                prediction_id = cursor.fetchone()['id']
            else:
                query = """
                    INSERT INTO predictions (input_id, model_version, prediction, confidence, latency)
                    VALUES (?, ?, ?, ?, ?)
                """
                cursor.execute(query, (input_id, model_version, prediction, confidence, latency))
                prediction_id = cursor.lastrowid
            
            self.connection.commit()
            cursor.close()
            
            logger.info(
                f"Prediction inserted successfully: ID={prediction_id}, "
                f"model={model_version}, confidence={confidence:.2f}"
            )
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error inserting prediction: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """
        Get recent prediction logs dari database
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List[Dict]: List of prediction records dengan user input
        """
        try:
            query = """
                SELECT 
                    p.id,
                    p.timestamp,
                    u.text_input,
                    p.model_version,
                    p.prediction,
                    p.confidence,
                    p.latency
                FROM predictions p
                JOIN users_inputs u ON p.input_id = u.id
                ORDER BY p.timestamp DESC
                LIMIT {}
            """.format('%s' if self.is_postgres else '?')
            
            results = self.execute_query(query, (limit,))
            logger.debug(f"Retrieved {len(results)} recent predictions")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving recent predictions: {e}")
            return []
    
    def get_dataset_snapshot(self, consent_only: bool = True) -> Any:
        """
        Get dataset snapshot untuk retraining
        
        Args:
            consent_only: Jika True, hanya ambil data dengan user consent
            
        Returns:
            pandas.DataFrame: Dataset snapshot
        """
        try:
            import pandas as pd
            
            query = """
                SELECT 
                    u.id,
                    u.timestamp,
                    u.text_input,
                    p.prediction,
                    p.confidence,
                    p.model_version
                FROM users_inputs u
                JOIN predictions p ON u.id = p.input_id
            """
            
            if consent_only:
                query += " WHERE u.user_consent = TRUE" if self.is_postgres else " WHERE u.user_consent = 1"
            
            query += " ORDER BY u.timestamp DESC"
            
            results = self.execute_query(query)
            df = pd.DataFrame(results)
            
            logger.info(f"Dataset snapshot retrieved: {len(df)} records, consent_only={consent_only}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving dataset snapshot: {e}")
            import pandas as pd
            return pd.DataFrame()
    
    def get_metrics_by_version(self) -> Dict[str, Dict]:
        """
        Get aggregated metrics per model version
        
        Returns:
            Dict[str, Dict]: Metrics per model version
        """
        try:
            query = """
                SELECT 
                    model_version,
                    COUNT(*) as prediction_count,
                    AVG(confidence) as avg_confidence,
                    AVG(latency) as avg_latency,
                    MIN(latency) as min_latency,
                    MAX(latency) as max_latency
                FROM predictions
                GROUP BY model_version
                ORDER BY model_version
            """
            results = self.execute_query(query)
            
            # Convert to nested dictionary
            metrics = {}
            for row in results:
                version = row['model_version']
                metrics[version] = {
                    'prediction_count': row['prediction_count'],
                    'avg_confidence': round(float(row['avg_confidence']), 4) if row['avg_confidence'] else 0,
                    'avg_latency': round(float(row['avg_latency']), 4) if row['avg_latency'] else 0,
                    'min_latency': round(float(row['min_latency']), 4) if row['min_latency'] else 0,
                    'max_latency': round(float(row['max_latency']), 4) if row['max_latency'] else 0
                }
            
            logger.debug(f"Metrics retrieved for {len(metrics)} model versions")
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving metrics by version: {e}")
            return {}
    
    def initialize_schema(self, schema_file: str = None) -> bool:
        """
        Initialize database schema dari SQL file
        
        Args:
            schema_file: Path ke schema SQL file (auto-detect jika None)
            
        Returns:
            bool: True jika initialization berhasil
        """
        try:
            if not self.connection:
                self.connect()
            
            # Check if tables already exist
            if self._tables_exist():
                logger.info("Database tables already exist, skipping initialization")
                return True
            
            # Auto-detect schema file
            if schema_file is None:
                schema_file = "database/schema_postgres.sql" if self.is_postgres else "database/schema.sql"
            
            # Read schema file
            schema_path = Path(schema_file)
            if not schema_path.exists():
                logger.error(f"Schema file not found: {schema_file}")
                return False
            
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            # Execute schema SQL
            cursor = self.connection.cursor()
            
            if self.is_postgres:
                # PostgreSQL: execute statements one by one
                statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
                for statement in statements:
                    if statement:
                        cursor.execute(statement)
            else:
                # SQLite: use executescript
                cursor.executescript(schema_sql)
            
            self.connection.commit()
            cursor.close()
            
            logger.info("Database schema initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def _tables_exist(self) -> bool:
        """
        Check apakah tables sudah exist
        
        Returns:
            bool: True jika tables exist
        """
        try:
            if self.is_postgres:
                query = """
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('users_inputs', 'predictions')
                """
            else:
                query = """
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('users_inputs', 'predictions')
                """
            
            results = self.execute_query(query)
            return len(results) == 2
            
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def migrate_schema(self, migration_sql: str) -> bool:
        """
        Execute database migration (future-proof untuk schema changes)
        
        Args:
            migration_sql: SQL statements untuk migration
            
        Returns:
            bool: True jika migration berhasil
        """
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            
            if self.is_postgres:
                statements = [s.strip() for s in migration_sql.split(';') if s.strip()]
                for statement in statements:
                    if statement:
                        cursor.execute(statement)
            else:
                cursor.executescript(migration_sql)
            
            self.connection.commit()
            cursor.close()
            
            logger.info("Database migration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error executing database migration: {e}")
            if self.connection:
                self.connection.rollback()
            return False
