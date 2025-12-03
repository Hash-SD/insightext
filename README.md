# InsightText - Sistem AI Berbasis Teks untuk Sentiment Analysis

Aplikasi MLOps dengan Streamlit untuk analisis sentimen teks menggunakan Naive Bayes dengan TF-IDF. Dilengkapi dengan model management system yang memungkinkan update model dengan archiving dan rollback.

**Status**: Production Ready ✅ | **Version**: 1.1 | **Last Updated**: December 2024

---

## 🚀 Features

- ✅ **Model v1**: Indonesian Sentiment Analysis (MBG YouTube dataset) - 69.72% accuracy
- ✅ **Model v2**: English Sentiment Analysis (IMDB dataset) - 86.47% accuracy
- ✅ Real-time predictions dengan confidence scores
- ✅ User consent management untuk data privacy
- ✅ PII detection dan anonymization
- ✅ Prediction history dan monitoring dashboard
- ✅ SQLite database (local) / PostgreSQL (production)
- ✅ **Model Management System**: Archive, update, validate, dan rollback models (NEW)
- ✅ **Dashboard Integration**: 4-tab model management interface (NEW)
- ✅ **Bug Fixes**: NoneType formatting error fixed in model_updater.py (NEW)
- ✅ Comprehensive logging dan error handling

---

## 📋 Prerequisites

- Python 3.8+
- Virtual Environment (recommended)

---

## 🛠️ Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd mlops
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy example env file
copy .env.example .env   # Windows
cp .env.example .env     # Linux/Mac
```

### 5. Run Application

```bash
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

---

## 📁 Project Structure

```
mlops/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest configuration
├── README.md                       # This file
├── DEPLOYMENT_CHECKLIST.md         # Deployment guide
├── config/
│   └── settings.py                # Application settings
├── database/
│   ├── db_manager.py              # Database operations
│   ├── db_manager_postgres.py      # PostgreSQL implementation
│   └── schema.sql                  # SQLite schema
├── models/
│   ├── model_loader.py            # Model loading logic
│   ├── model_archiver.py           # Archive management (NEW)
│   ├── model_updater.py            # Update orchestration (NEW)
│   ├── preprocessor.py             # Text preprocessing
│   ├── placeholders.py             # Model placeholders
│   └── saved_model/                # v1 Indonesian model
│       ├── archived/               # Old model versions (NEW)
│       └── logs/model_updates/     # Update reports (NEW)
├── services/
│   ├── prediction_service.py       # Prediction logic
│   ├── monitoring_service.py       # Metrics & monitoring
│   └── retraining_service.py       # Retraining pipeline
├── ui/
│   ├── main_area.py                # Main UI components
│   ├── sidebar.py                  # Sidebar components
│   └── monitoring.py               # Monitoring dashboard + model management (UPDATED)
├── utils/
│   ├── logger.py                   # Logging utility
│   ├── privacy.py                  # PII detection
│   └── validators.py               # Input validation
└── tests/
    ├── test_integration.py         # End-to-end tests
    ├── test_model_updater_bug_fix.py # Bug fix tests (NEW)
    ├── test_database/              # Database layer tests
    └── test_services/              # Service layer tests
```

---

## 🔧 Model Versions

| Version | Language | Dataset | Accuracy | Status |
|---------|----------|---------|----------|--------|
| v1 | Indonesian | MBG YouTube | 69.72% | Current (updateable) |
| v2 | English | IMDB | 86.47% | Current |

---

## 📊 Basic Usage

1. **Select Model Version**: Choose v1 (Indonesian) or v2 (English) from sidebar
2. **Enter Text**: Input your text for sentiment analysis
3. **User Consent**: Toggle if you allow storing your data
4. **Analyze**: Click button to get prediction
5. **View Results**: See sentiment, confidence, and processing time

---

## 🚀 Model Management System (NEW)

Access via: **Monitoring Dashboard → Model Management Tab**

### Update Model
```
1. Train new model dengan balanced data
2. Go to: 📊 Monitoring → 🚀 Model Management → 📤 Update Model
3. Upload model files (model_pipeline.pkl + preprocessor.pkl)
4. Input metrics (accuracy, F1 score, training samples)
5. Add update reason (e.g., "Trained with balanced data")
6. Click: 🚀 Update Model Sekarang

System akan automatically:
- ✓ Validate model structure
- ✓ Validate model performance
- ✓ Archive old model dengan metadata
- ✓ Deploy new model ke production
- ✓ Generate comparison report
```

### Archive Management
- View all archived model versions dengan timestamps
- See metrics dari setiap version
- Restore atau delete archives

### Model Comparison
- Compare current model dengan archived versions
- See accuracy improvements/degradation
- Track update history

### 3-Step Model Update

#### Step 1: Train Balanced Model
```bash
python naive_bayes_full.py
# Outputs: balanced_sentiment_model.pkl, tfidf_vectorizer.pkl
```

#### Step 2: Update via Dashboard
1. Navigate to: Monitoring → Model Management → Update Model
2. Upload model files
3. Input metrics (accuracy, F1, samples)
4. Click: Update Model Sekarang

#### Step 3: Verify
1. Check Overview tab untuk new metrics
2. Compare old vs new di Comparison tab
3. View archive di Archive Management tab

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_database/ -v        # Database tests
pytest tests/test_services/ -v        # Service layer tests
pytest tests/test_integration.py -v   # Integration tests
pytest tests/test_model_updater_bug_fix.py -v  # Bug fix tests (NEW)

# Run with coverage
pytest --cov=. --cov-report=html tests/

# Run without Supabase tests (if no cloud connection)
pytest tests/ --ignore=tests/test_database/test_db_manager_supabase.py -v
```

**Current Test Status**: ✅ **103 passed, 28 skipped, 0 failed**

---

## 🔐 Privacy

- PII (email, phone, ID numbers) are automatically detected and anonymized
- User consent is required before storing data
- All data is stored securely in the database

---

## 🐛 Recent Improvements & Bug Fixes

### Version 1.1 - Model Management & Bug Fixes

#### New Features
- **ModelArchiver**: Archive models with metadata preservation
  - Stores old models with timestamps
  - Preserves model metrics dan metadata
  - Easy restore and management

- **ModelUpdater**: Orchestrate model updates dengan validation
  - Automatic validation sebelum deployment
  - Performance checking (minimum accuracy 60%, F1 50%)
  - Structure validation
  - Comprehensive logging

- **ModelUpdateValidator**: Validate model structure dan performance
  - Checks required files (model_pipeline.pkl, preprocessor.pkl)
  - Validates metrics against minimum thresholds
  - Tests prediction functionality

- **Dashboard Integration**: 4-tab model management interface
  - Overview: Current model metrics
  - Update Model: Upload dan configure new model
  - Archive Management: View dan manage old versions
  - Model Comparison: Compare performance metrics

#### Bug Fixes
- **Fixed**: NoneType formatting error in `models/model_updater.py` (lines 310-314)
  - **Issue**: Model update crashed during logging when metrics missing 'accuracy' key
  - **Root Cause**: Code assumed `.get('accuracy')` would always return numeric value
  - **Solution**: Added defensive null-checks with default values (0.0)
  - **Impact**: Update process now handles missing metrics gracefully
  - **Testing**: 4 new test cases covering edge cases + full regression suite (103/103 passing)

---

## 📝 Deployment Checklist

- [x] ModelArchiver class created (350+ lines)
- [x] ModelUpdater class created (400+ lines)
- [x] ModelUpdateValidator class created
- [x] Dashboard integration completed (400+ new lines)
- [x] Archive directory structure created
- [x] Update logs directory ready
- [x] Helper script created (update_model_v1_balanced.py)
- [x] All tests passing (103/103)
- [x] Bug fixes implemented and verified
- [x] Code reviewed and validated
- [x] No breaking changes to existing code
- [x] Full backward compatibility maintained

**Status**: ✅ **Ready for Production**

---

## 📋 Detailed Model Update Guide

### API Usage

#### ModelArchiver
```python
from models.model_archiver import ModelArchiver

archiver = ModelArchiver(archive_base_path='models/archived')

# Archive current model
archive_path = archiver.archive_model(
    version='v1',
    current_model_path='models/saved_model',
    metrics={'accuracy': 0.6972, 'f1_score': 0.6782},
    notes='Original model before balanced data update'
)

# List all archived models
archives = archiver.list_archived_models(version='v1')

# Restore from archive
success = archiver.restore_model(
    archive_path='models/archived/v1_20240102_120000',
    restore_to_path='models/saved_model'
)
```

#### ModelUpdater
```python
from models.model_updater import ModelUpdater

updater = ModelUpdater()

# Update dengan validation
success, report = updater.update_model_v1(
    new_model_path='path/to/new/model',
    new_metrics={'accuracy': 0.75, 'f1_score': 0.73},
    update_reason='Trained with balanced data',
    auto_validate=True
)

# Rollback to previous version
success, result = updater.rollback_to_archive('models/archived/v1_20240102_120000')

# View update history
history = updater.list_update_history(limit=10)
```

### Validation Rules

- **Minimum Accuracy**: 60%
- **Minimum F1 Score**: 50%
- **Required Files**: model_pipeline.pkl, preprocessor.pkl
- **Optional File**: training_config.json

---

## 🔗 Additional References

### Documentation Files
- **DEPLOYMENT_CHECKLIST.md**: Step-by-step deployment guide
- **tests/README.md**: Comprehensive testing documentation
- **models/model_archiver.py**: Archive system implementation
- **models/model_updater.py**: Update orchestration implementation

### Key Directories
- **models/saved_model/**: Current production model
- **models/archived/**: Archive of previous model versions
- **logs/model_updates/**: Update operation logs and reports

---

## 📝 License

MIT License

---

## 👥 Contributors

- MLOps Team

---

**Version**: 1.1 (Model Management System + Bug Fixes)  
**Status**: ✅ Production Ready  
**Last Updated**: December 2024  
**Test Coverage**: 103 passing, 28 skipped, 0 failed
