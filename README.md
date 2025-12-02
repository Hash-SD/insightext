# Sistem AI Berbasis Teks - Sentiment Analysis

Aplikasi MLOps dengan Streamlit untuk analisis sentimen teks menggunakan Naive Bayes dengan TF-IDF.

## 🚀 Features

- ✅ **Model v1**: Indonesian Sentiment Analysis (MBG YouTube dataset) - 69.72% accuracy
- ✅ **Model v2**: English Sentiment Analysis (IMDB dataset) - 86.47% accuracy
- ✅ Real-time predictions dengan confidence scores
- ✅ User consent management untuk data privacy
- ✅ PII detection dan anonymization
- ✅ Prediction history dan monitoring dashboard
- ✅ SQLite database (local) / PostgreSQL (production)
- ✅ Comprehensive logging dan error handling

## 📋 Prerequisites

- Python 3.8+
- Virtual Environment (recommended)

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

## 📁 Project Structure

```
mlops/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── config/
│   └── settings.py       # Application settings
├── database/
│   ├── db_manager.py     # Database operations
│   └── schema.sql        # SQLite schema
├── models/
│   ├── model_loader.py   # Model loading logic
│   ├── naive_bayes_loader.py  # Naive Bayes model
│   ├── text_preprocessor.py   # Text preprocessing
│   └── saved_model/      # v1 Indonesian model
├── services/
│   ├── prediction_service.py   # Prediction logic
│   ├── monitoring_service.py   # Metrics & monitoring
│   └── retraining_service.py   # Retraining pipeline
├── ui/
│   ├── main_area.py      # Main UI components
│   ├── sidebar.py        # Sidebar components
│   └── monitoring.py     # Monitoring dashboard
├── utils/
│   ├── logger.py         # Logging utility
│   ├── privacy.py        # PII detection
│   └── validators.py     # Input validation
└── tests/                # Unit & integration tests
```

## 🔧 Model Versions

| Version | Language | Dataset | Accuracy | Labels |
|---------|----------|---------|----------|--------|
| v1 | Indonesian | MBG YouTube | 69.72% | negatif, netral, positif |
| v2 | English | IMDB | 86.47% | negative, positive |

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run without Supabase tests (if no cloud connection)
pytest tests/ --ignore=tests/test_database/test_db_manager_supabase.py -v
```

## 📊 Usage

1. **Select Model Version**: Choose v1 (Indonesian) or v2 (English) from sidebar
2. **Enter Text**: Input your text for sentiment analysis
3. **User Consent**: Toggle if you allow storing your data
4. **Analyze**: Click button to get prediction
5. **View Results**: See sentiment, confidence, and processing time

## 🔐 Privacy

- PII (email, phone, ID numbers) are automatically detected and anonymized
- User consent is required before storing data
- All data is stored securely in the database

## 📝 License

MIT License

## 👥 Contributors

- MLOps Team
