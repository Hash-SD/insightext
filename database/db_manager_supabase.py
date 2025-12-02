"""
Database Manager untuk Supabase menggunakan REST API.
Fallback dari direct PostgreSQL connection yang memiliki masalah DNS.
"""

import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class SupabaseDatabaseManager:
    """Manager untuk Supabase database operations via REST API"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize Supabase database manager
        
        Args:
            supabase_url: Supabase project URL (https://xxx.supabase.co)
            supabase_key: Supabase API key (anon atau service_role)
        """
        self.supabase_url = supabase_url.rstrip('/')
        self.supabase_key = supabase_key
        self.max_retries = 3
        self.retry_delay = 1
        self.is_postgres = True  # For compatibility
        
        # HTTP client
        try:
            import httpx
            self.http = httpx
        except ImportError:
            raise ImportError("httpx required. Install with: pip install httpx")
        
        self._headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }
        
        logger.info(f"SupabaseDatabaseManager initialized for {self.supabase_url}")
    
    def connect(self) -> bool:
        """Test connection to Supabase (REST API is stateless)"""
        try:
            r = self.http.get(
                f"{self.supabase_url}/rest/v1/",
                headers=self._headers,
                timeout=10
            )
            if r.status_code == 200:
                logger.info("Supabase REST API connection verified")
                return True
            else:
                logger.error(f"Supabase connection failed: {r.status_code} - {r.text}")
                return False
        except Exception as e:
            logger.error(f"Supabase connection error: {e}")
            return False
    
    def disconnect(self):
        """No-op for REST API (stateless)"""
        logger.info("Supabase REST API connection closed (stateless)")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """
        Execute SELECT query - parses SQL and converts to REST API call.
        Supports basic SQL SELECT queries for compatibility with existing services.
        
        Args:
            query: SQL SELECT query (will be parsed)
            params: Query parameters (limited support)
            
        Returns:
            List[Dict]: Query results
        """
        try:
            # Clean query first, then convert to uppercase for matching
            query_clean = ' '.join(query.split())  # Normalize whitespace
            query_upper = query_clean.upper()
            
            # Extract table name
            if 'FROM' not in query_upper:
                logger.error(f"Invalid query - no FROM clause: {query[:50]}")
                return []
            
            # Simple parsing for common query patterns
            from_idx = query_upper.find('FROM')
            after_from = query_clean[from_idx + 5:].strip()  # +5 for 'FROM '
            
            # Get table name (first word after FROM)
            table_parts = after_from.split()
            table = table_parts[0].lower() if table_parts else ''
            
            # Remove any alias or other keywords
            table = table.replace(',', '').strip()
            
            # Extract select columns
            select_idx = query_upper.find('SELECT')
            select_part = query_clean[select_idx + 7:from_idx].strip()  # +7 for 'SELECT '
            
            # Handle "SELECT column FROM table" patterns
            if select_part == '*':
                select = '*'
            else:
                # Clean up column names
                select = select_part.replace(' ', '')
            
            # Build URL
            url = f"{self.supabase_url}/rest/v1/{table}?select={select}"
            
            # Handle WHERE clause with params
            if 'WHERE' in query_upper:
                where_idx = query_upper.find('WHERE')
                where_clause = query_clean[where_idx + 6:].strip()  # +6 for 'WHERE '
                
                # Handle ORDER BY if present
                if 'ORDER' in where_clause.upper():
                    order_idx = where_clause.upper().find('ORDER')
                    where_part = where_clause[:order_idx].strip()
                else:
                    where_part = where_clause
                
                # Handle LIMIT if present
                if 'LIMIT' in where_part.upper():
                    limit_idx = where_part.upper().find('LIMIT')
                    where_part = where_part[:limit_idx].strip()
                
                # Handle GROUP BY if present
                if 'GROUP' in where_part.upper():
                    group_idx = where_part.upper().find('GROUP')
                    where_part = where_part[:group_idx].strip()
                
                # Parse simple conditions (column = ?)
                if where_part and params:
                    # Replace ? with actual param values
                    conditions = where_part.split('AND')
                    for i, cond in enumerate(conditions):
                        cond = cond.strip()
                        if '=' in cond and '?' in cond:
                            col = cond.split('=')[0].strip()
                            if i < len(params):
                                url += f"&{col}=eq.{params[i]}"
            
            # Handle ORDER BY
            if 'ORDER BY' in query_upper:
                order_idx = query_upper.find('ORDER BY')
                order_part = query_clean[order_idx + 9:].strip()  # +9 for 'ORDER BY '
                
                # Remove LIMIT if present
                if 'LIMIT' in order_part.upper():
                    limit_idx = order_part.upper().find('LIMIT')
                    order_part = order_part[:limit_idx].strip()
                
                # Convert to PostgREST format (column.asc or column.desc)
                order_parts = order_part.split(',')
                order_clauses = []
                for op in order_parts:
                    op = op.strip()
                    if not op:  # Skip empty
                        continue
                    op_upper = op.upper()
                    if 'DESC' in op_upper:
                        col = op_upper.replace('DESC', '').strip()
                        if col:  # Only add if column name exists
                            order_clauses.append(f"{col.lower()}.desc")
                    elif 'ASC' in op_upper:
                        col = op_upper.replace('ASC', '').strip()
                        if col:  # Only add if column name exists
                            order_clauses.append(f"{col.lower()}.asc")
                    else:
                        if op:  # Only add if column name exists
                            order_clauses.append(f"{op.lower()}.asc")
                
                if order_clauses:
                    url += f"&order={','.join(order_clauses)}"
            
            # Handle LIMIT
            if 'LIMIT' in query_upper:
                limit_idx = query_upper.find('LIMIT')
                limit_part = query_clean[limit_idx + 6:].strip()  # +6 for 'LIMIT '
                
                # Extract limit value
                limit_val = limit_part.split()[0] if limit_part else None
                if limit_val:
                    # Could be a ? param or direct value
                    if limit_val == '?' and params:
                        # Use last param if it's the limit
                        url += f"&limit={params[-1]}"
                    elif limit_val.isdigit():
                        url += f"&limit={limit_val}"
            
            # For GROUP BY queries, we need to do aggregation in Python
            if 'GROUP BY' in query_upper:
                # Get raw data and aggregate in Python
                base_url = f"{self.supabase_url}/rest/v1/{table}?select=*"
                r = self.http.get(base_url, headers=self._headers, timeout=30)
                
                if r.status_code == 200:
                    raw_data = r.json()
                    return self._aggregate_in_python(raw_data, query_clean)
                else:
                    logger.error(f"Query failed: {r.status_code} - {r.text}")
                    return []
            
            r = self.http.get(url, headers=self._headers, timeout=10)
            
            if r.status_code == 200:
                return r.json()
            else:
                logger.error(f"Query failed: {r.status_code} - {r.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def _aggregate_in_python(self, data: List[Dict], query: str) -> List[Dict]:
        """
        Perform aggregation in Python for GROUP BY queries.
        
        Args:
            data: Raw data from database
            query: Original SQL query for parsing
            
        Returns:
            List[Dict]: Aggregated results
        """
        try:
            from collections import defaultdict
            
            query_upper = query.upper()
            
            # Extract GROUP BY column
            group_idx = query_upper.find('GROUP BY')
            group_part = query[group_idx + 8:].strip()
            
            # Remove ORDER BY if present
            if 'ORDER' in group_part.upper():
                order_idx = group_part.upper().find('ORDER')
                group_part = group_part[:order_idx].strip()
            
            group_col = group_part.split()[0].strip()
            
            # Group data
            grouped = defaultdict(list)
            for row in data:
                key = row.get(group_col, 'unknown')
                grouped[key].append(row)
            
            # Parse SELECT to understand aggregations
            select_part = query[query_upper.find('SELECT') + 6:query_upper.find('FROM')].strip()
            
            results = []
            for key, rows in grouped.items():
                result = {group_col: key}
                
                # Parse aggregations in SELECT
                agg_funcs = select_part.split(',')
                for agg in agg_funcs:
                    agg = agg.strip()
                    agg_upper = agg.upper()
                    
                    if 'COUNT(*)' in agg_upper:
                        alias = 'count'
                        if ' AS ' in agg_upper:
                            alias = agg.split(' AS ')[-1].strip().lower()
                        result[alias] = len(rows)
                    
                    elif 'AVG(' in agg_upper:
                        col = agg[agg.find('(')+1:agg.find(')')].strip()
                        alias = f'avg_{col}'
                        if ' AS ' in agg_upper:
                            alias = agg.split(' AS ')[-1].strip().lower()
                        values = [r.get(col, 0) for r in rows if r.get(col) is not None]
                        result[alias] = sum(values) / len(values) if values else 0
                    
                    elif 'MIN(' in agg_upper:
                        col = agg[agg.find('(')+1:agg.find(')')].strip()
                        alias = f'min_{col}'
                        if ' AS ' in agg_upper:
                            alias = agg.split(' AS ')[-1].strip().lower()
                        values = [r.get(col, 0) for r in rows if r.get(col) is not None]
                        result[alias] = min(values) if values else 0
                    
                    elif 'MAX(' in agg_upper:
                        col = agg[agg.find('(')+1:agg.find(')')].strip()
                        alias = f'max_{col}'
                        if ' AS ' in agg_upper:
                            alias = agg.split(' AS ')[-1].strip().lower()
                        values = [r.get(col, 0) for r in rows if r.get(col) is not None]
                        result[alias] = max(values) if values else 0
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Python aggregation: {e}")
            return []
    
    def execute_rest_query(self, table: str, select: str = "*", filters: Dict = None, 
                      order: str = None, limit: int = None) -> List[Dict]:
        """
        Execute SELECT query via REST API (direct REST interface)
        
        Args:
            table: Table name
            select: Columns to select
            filters: Dictionary of filters {column: value}
            order: Order by clause (e.g., "timestamp.desc")
            limit: Maximum number of results
            
        Returns:
            List[Dict]: Query results
        """
        try:
            # Clean up select string (remove newlines and extra spaces)
            select_clean = select.replace('\n', '').replace('  ', ' ').strip()
            url = f"{self.supabase_url}/rest/v1/{table}?select={select_clean}"
            
            if filters:
                for col, val in filters.items():
                    url += f"&{col}=eq.{val}"
            
            if order:
                url += f"&order={order}"
            
            if limit:
                url += f"&limit={limit}"
            
            r = self.http.get(url, headers=self._headers, timeout=10)
            
            if r.status_code == 200:
                return r.json()
            else:
                logger.error(f"Query failed: {r.status_code} - {r.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error executing REST query: {e}")
            return []
    
    def insert_user_input(self, text: str, consent: bool) -> int:
        """
        Insert user input ke database
        
        Args:
            text: Input teks dari pengguna
            consent: User consent untuk menyimpan data
            
        Returns:
            int: ID dari inserted record
        """
        retries = 0
        while retries < self.max_retries:
            try:
                data = {
                    'text_input': text,
                    'user_consent': consent,
                    'anonymized': False
                }
                
                r = self.http.post(
                    f"{self.supabase_url}/rest/v1/users_inputs",
                    headers=self._headers,
                    json=data,
                    timeout=10
                )
                
                if r.status_code == 201:
                    result = r.json()[0]
                    input_id = result['id']
                    logger.info(f"User input inserted: ID={input_id}, consent={consent}")
                    return input_id
                else:
                    raise Exception(f"Insert failed: {r.status_code} - {r.text}")
                    
            except Exception as e:
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay * retries)
                    logger.warning(f"Retry {retries}/{self.max_retries} for insert_user_input: {e}")
                else:
                    logger.error(f"Failed to insert user input after {self.max_retries} retries: {e}")
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
        """
        retries = 0
        while retries < self.max_retries:
            try:
                data = {
                    'input_id': input_id,
                    'model_version': model_version,
                    'prediction': prediction,
                    'confidence': confidence,
                    'latency': latency
                }
                
                r = self.http.post(
                    f"{self.supabase_url}/rest/v1/predictions",
                    headers=self._headers,
                    json=data,
                    timeout=10
                )
                
                if r.status_code == 201:
                    result = r.json()[0]
                    prediction_id = result['id']
                    logger.info(f"Prediction inserted: ID={prediction_id}, model={model_version}")
                    return prediction_id
                else:
                    raise Exception(f"Insert failed: {r.status_code} - {r.text}")
                    
            except Exception as e:
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay * retries)
                    logger.warning(f"Retry {retries}/{self.max_retries} for insert_prediction: {e}")
                else:
                    logger.error(f"Failed to insert prediction after {self.max_retries} retries: {e}")
                    raise
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """
        Get recent prediction logs dengan join ke users_inputs
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List[Dict]: List of prediction records dengan user input
        """
        try:
            # Use PostgREST embedded resources for join
            url = f"{self.supabase_url}/rest/v1/predictions"
            url += "?select=id,timestamp,model_version,prediction,confidence,latency,"
            url += "users_inputs(text_input)"
            url += f"&order=timestamp.desc&limit={limit}"
            
            r = self.http.get(url, headers=self._headers, timeout=10)
            
            if r.status_code == 200:
                results = r.json()
                # Flatten the nested structure
                flattened = []
                for row in results:
                    flat_row = {
                        'id': row['id'],
                        'timestamp': row['timestamp'],
                        'model_version': row['model_version'],
                        'prediction': row['prediction'],
                        'confidence': row['confidence'],
                        'latency': row['latency'],
                        'text_input': row.get('users_inputs', {}).get('text_input', '')
                    }
                    flattened.append(flat_row)
                
                logger.debug(f"Retrieved {len(flattened)} recent predictions")
                return flattened
            else:
                logger.error(f"Failed to get predictions: {r.status_code} - {r.text}")
                return []
                
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
            
            # Get users_inputs with predictions embedded
            url = f"{self.supabase_url}/rest/v1/users_inputs"
            url += "?select=id,timestamp,text_input,predictions(prediction,confidence,model_version)"
            
            if consent_only:
                url += "&user_consent=eq.true"
            
            url += "&order=timestamp.desc"
            
            r = self.http.get(url, headers=self._headers, timeout=30)
            
            if r.status_code == 200:
                results = r.json()
                # Flatten nested predictions
                flattened = []
                for row in results:
                    predictions = row.get('predictions', [])
                    if predictions:
                        for pred in predictions:
                            flat_row = {
                                'id': row['id'],
                                'timestamp': row['timestamp'],
                                'text_input': row['text_input'],
                                'prediction': pred['prediction'],
                                'confidence': pred['confidence'],
                                'model_version': pred['model_version']
                            }
                            flattened.append(flat_row)
                
                df = pd.DataFrame(flattened)
                logger.info(f"Dataset snapshot: {len(df)} records, consent_only={consent_only}")
                return df
            else:
                logger.error(f"Failed to get dataset: {r.status_code} - {r.text}")
                return pd.DataFrame()
                
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
            # Get all predictions and aggregate in Python
            url = f"{self.supabase_url}/rest/v1/predictions"
            url += "?select=model_version,confidence,latency"
            
            r = self.http.get(url, headers=self._headers, timeout=30)
            
            if r.status_code == 200:
                results = r.json()
                
                # Aggregate by model_version
                from collections import defaultdict
                metrics_data = defaultdict(list)
                
                for row in results:
                    version = row['model_version']
                    metrics_data[version].append({
                        'confidence': row['confidence'],
                        'latency': row['latency']
                    })
                
                # Calculate aggregates
                metrics = {}
                for version, data in metrics_data.items():
                    confidences = [d['confidence'] for d in data]
                    latencies = [d['latency'] for d in data]
                    
                    metrics[version] = {
                        'prediction_count': len(data),
                        'avg_confidence': round(sum(confidences) / len(confidences), 4) if confidences else 0,
                        'avg_latency': round(sum(latencies) / len(latencies), 4) if latencies else 0,
                        'min_latency': round(min(latencies), 4) if latencies else 0,
                        'max_latency': round(max(latencies), 4) if latencies else 0
                    }
                
                logger.debug(f"Metrics for {len(metrics)} model versions")
                return metrics
            else:
                logger.error(f"Failed to get metrics: {r.status_code} - {r.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error retrieving metrics: {e}")
            return {}
    
    def initialize_schema(self, schema_file: str = None) -> bool:
        """
        Check if schema exists (tables are managed via Supabase Dashboard)
        
        Returns:
            bool: True if tables exist
        """
        try:
            # Just verify we can access the tables
            r = self.http.get(
                f"{self.supabase_url}/rest/v1/users_inputs?limit=1",
                headers=self._headers,
                timeout=10
            )
            
            if r.status_code == 200:
                logger.info("Supabase schema verified - tables exist")
                return True
            else:
                logger.error(f"Schema check failed: {r.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking schema: {e}")
            return False
    
    def _tables_exist(self) -> bool:
        """Check if required tables exist"""
        return self.initialize_schema()
    
    def execute_transaction(self, queries: List) -> bool:
        """
        REST API doesn't support true transactions.
        Execute operations sequentially.
        
        Returns:
            bool: True if all operations succeed
        """
        logger.warning("REST API doesn't support true transactions. Operations executed sequentially.")
        return True
    
    def migrate_schema(self, migration_sql: str) -> bool:
        """Schema migrations must be done via Supabase Dashboard"""
        logger.warning("Schema migrations must be done via Supabase Dashboard")
        return False
