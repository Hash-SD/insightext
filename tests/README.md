# Tests Documentation

Dokumentasi lengkap untuk testing MLOps Streamlit Text AI application.

## Test Structure

```
tests/
├── __init__.py
├── README.md                           # Dokumentasi ini
├── MANUAL_TESTING_CHECKLIST.md        # Checklist untuk manual testing
├── test_database/                      # Database layer tests
│   ├── __init__.py
│   ├── test_db_manager.py             # DatabaseManager tests (SQLite)
│   └── test_db_manager_supabase.py    # DatabaseManager tests (Supabase/PostgreSQL)
├── test_services/                      # Services layer tests
│   ├── __init__.py
│   ├── test_prediction_service.py     # PredictionService tests
│   ├── test_monitoring_service.py     # MonitoringService tests
│   └── test_retraining_service.py     # RetrainingService tests
└── test_integration.py                 # End-to-end integration tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov --cov-report=html
```

### Run Specific Test Categories

```bash
# Run database tests only (SQLite)
pytest tests/test_database/test_db_manager.py

# Run Supabase tests only
pytest tests/test_database/test_db_manager_supabase.py

# Run all database tests (SQLite + Supabase)
pytest tests/test_database/

# Run service tests only
pytest tests/test_services/

# Run integration tests only
pytest tests/test_integration.py

# Run specific test file
pytest tests/test_database/test_db_manager.py

# Run specific test class
pytest tests/test_database/test_db_manager.py::TestDatabaseConnection

# Run specific test function
pytest tests/test_database/test_db_manager.py::TestDatabaseConnection::test_connect_success
```

### Run Tests with Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only database tests
pytest -m database
```

## Test Coverage

### Current Coverage

- **Database Layer**: 
  - SQLite Tests: 22 tests covering connection, schema, insert, query, and transaction operations
  - Supabase Tests: 30+ tests covering PostgreSQL-specific features, foreign keys, performance, and data integrity
- **Services Layer**: 
  - PredictionService: 15+ tests covering validation, prediction flow, logging, and error handling
  - MonitoringService: 15+ tests covering metrics, latency, drift detection, and monitoring
  - RetrainingService: 20+ tests covering retraining pipeline, dataset handling, and validation
- **Integration Tests**: 20+ tests covering end-to-end flows and system integration

### Generate Coverage Report

```bash
# Generate HTML coverage report
pytest --cov --cov-report=html

# Open coverage report
# Windows: start htmlcov/index.html
# Mac/Linux: open htmlcov/index.html
```

## Test Categories

### 1. Unit Tests

Unit tests test individual components in isolation dengan mocked dependencies.

**Database Tests**:

*SQLite Tests* (`test_database/test_db_manager.py`):
- Connection and retry logic
- Schema initialization
- Insert operations (user input, predictions)
- Query operations (recent predictions, dataset snapshot, metrics)
- Transaction operations
- Error handling

*Supabase/PostgreSQL Tests* (`test_database/test_db_manager_supabase.py`):
- Real Supabase connection testing
- PostgreSQL-specific features (foreign keys, constraints)
- Schema structure validation
- Insert operations dengan real database
- Query operations dan filtering
- Transaction commit/rollback
- Performance testing (bulk inserts, query speed)
- Connection resilience
- Data integrity (timestamps, encoding, data types)

**Service Tests** (`test_services/`):
- PredictionService: validation, prediction flow, logging
- MonitoringService: metrics aggregation, latency distribution, drift detection
- RetrainingService: dataset handling, training, evaluation, MLflow logging

### 2. Integration Tests

Integration tests test complete flows dengan real components.

**End-to-End Flows** (`test_integration.py`):
- Complete prediction flow: input → prediction → database → display
- Model version switching
- Monitoring integration dengan real data
- Retraining pipeline dengan real data
- Data consistency across operations
- Error recovery and resilience
- Concurrent operations

### 3. Manual Tests

Manual testing checklist untuk UI/UX testing.

**Checklist** (`MANUAL_TESTING_CHECKLIST.md`):
- UI responsiveness
- Input testing
- Button functionality
- Consent checkbox behavior
- Model version selector
- Prediction results display
- Monitoring dashboard
- Error scenarios
- Bahasa Indonesia verification

## Writing New Tests

### Test Structure

```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_dependency():
    """Create mock dependency"""
    mock = Mock()
    # Setup mock behavior
    return mock

@pytest.fixture
def test_subject(mock_dependency):
    """Create test subject dengan mocked dependencies"""
    return TestSubject(mock_dependency)

class TestFeature:
    """Test specific feature"""
    
    def test_success_case(self, test_subject):
        """Test successful operation"""
        result = test_subject.do_something()
        assert result is not None
    
    def test_error_case(self, test_subject):
        """Test error handling"""
        with pytest.raises(Exception):
            test_subject.do_invalid_operation()
```

### Best Practices

1. **Use Descriptive Names**: Test names should clearly describe what they test
2. **One Assertion Per Test**: Focus each test on one specific behavior
3. **Use Fixtures**: Reuse setup code dengan pytest fixtures
4. **Mock External Dependencies**: Isolate unit tests dari external systems
5. **Test Edge Cases**: Include tests untuk boundary conditions dan error scenarios
6. **Keep Tests Fast**: Unit tests should run quickly (< 1 second each)
7. **Clean Up Resources**: Use fixtures untuk setup dan teardown

### Test Naming Convention

```python
def test_<function_name>_<scenario>_<expected_result>():
    """
    Example:
    - test_predict_valid_input_returns_result()
    - test_predict_invalid_input_raises_error()
    - test_connect_database_error_retries()
    """
    pass
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Supabase Testing

### Prerequisites

Supabase tests memerlukan:
1. Valid `DATABASE_URL` di file `.env`
2. Database URL harus PostgreSQL format: `postgresql://...`
3. Koneksi internet ke Supabase

### Running Supabase Tests

```bash
# Run all Supabase tests
pytest tests/test_database/test_db_manager_supabase.py -v

# Run specific Supabase test class
pytest tests/test_database/test_db_manager_supabase.py::TestSupabaseConnection -v

# Run dengan output detail
pytest tests/test_database/test_db_manager_supabase.py -v -s
```

### Supabase Test Categories

1. **Connection Tests**: Verify connection ke Supabase PostgreSQL
2. **Schema Tests**: Validate table structure, foreign keys, indexes
3. **Insert Tests**: Test data insertion dengan foreign key constraints
4. **Query Tests**: Test data retrieval dan filtering
5. **Transaction Tests**: Test commit/rollback behavior
6. **Performance Tests**: Measure insert dan query performance
7. **Resilience Tests**: Test reconnection dan error handling
8. **Data Integrity Tests**: Test timestamps, encoding, data types

### Important Notes

⚠️ **WARNING**: Supabase tests connect to REAL database!

- Tests akan create dan cleanup test data
- Gunakan dengan hati-hati di production environment
- Test data akan di-cleanup setelah each test
- Jika cleanup gagal, data mungkin tertinggal di database

### Skipping Supabase Tests

Supabase tests akan automatically skip jika:
- `DATABASE_URL` tidak di-set di `.env`
- `DATABASE_URL` bukan PostgreSQL format
- Connection ke Supabase gagal

```bash
# Skip message akan muncul:
# "Supabase DATABASE_URL not configured or not PostgreSQL"
```

### Cleanup Test Data

Test data di-cleanup automatically menggunakan timestamp:
- Setiap test track start time
- Setelah test, delete semua data created setelah start time
- Foreign key constraints di-handle (delete predictions first, then inputs)

Manual cleanup jika diperlukan:
```sql
-- Connect ke Supabase dan run:
DELETE FROM predictions WHERE timestamp >= '2024-01-01';
DELETE FROM users_inputs WHERE timestamp >= '2024-01-01';
```

### Troubleshooting Supabase Tests

**Issue: Tests skipped dengan "DATABASE_URL not configured"**
```bash
# Solution: Check .env file
cat .env | grep DATABASE_URL

# Ensure format is correct:
# DATABASE_URL=postgresql://user:pass@host:port/database
```

**Issue: Connection timeout**
```bash
# Solution: Check internet connection dan Supabase status
# Verify DATABASE_URL credentials are correct
```

**Issue: Foreign key violation errors**
```bash
# Solution: This is expected behavior in PostgreSQL
# Tests verify that foreign key constraints are enforced
```

**Issue: Test data not cleaned up**
```bash
# Solution: Manually cleanup using SQL above
# Or run cleanup script:
python -c "from database.db_manager import DatabaseManager; \
           db = DatabaseManager(os.getenv('DATABASE_URL')); \
           db.connect(); \
           # Add cleanup queries here"
```

## Troubleshooting

### Common Issues

**Issue: Tests fail dengan "ModuleNotFoundError"**
```bash
# Solution: Install package in development mode
pip install -e .
```

**Issue: Database tests fail dengan "database is locked"**
```bash
# Solution: Ensure previous test cleanup completed
# Check for orphaned database connections
```

**Issue: Integration tests are slow**
```bash
# Solution: Run only unit tests during development
pytest tests/test_database/ tests/test_services/
```

**Issue: Mock not working as expected**
```bash
# Solution: Verify mock is patched at correct location
# Use pytest-mock for easier mocking
```

## Test Maintenance

### When to Update Tests

- When adding new features
- When fixing bugs (add regression test)
- When refactoring code
- When requirements change

### Test Review Checklist

- [ ] All tests pass
- [ ] New features have tests
- [ ] Edge cases covered
- [ ] Error handling tested
- [ ] Documentation updated
- [ ] Coverage maintained or improved

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-mock Documentation](https://pytest-mock.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)

## Contact

Untuk pertanyaan atau issues terkait testing, silakan buat issue di repository atau hubungi tim development.
