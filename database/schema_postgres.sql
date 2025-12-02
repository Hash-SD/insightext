-- Schema untuk Sistem AI Berbasis Teks (PostgreSQL/Supabase)
-- Database tables untuk menyimpan user inputs dan predictions

-- Table untuk menyimpan input teks dari pengguna
CREATE TABLE IF NOT EXISTS users_inputs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    text_input TEXT NOT NULL,
    user_consent BOOLEAN DEFAULT FALSE,
    anonymized BOOLEAN DEFAULT FALSE
);

-- Table untuk menyimpan hasil prediksi
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    input_id INTEGER NOT NULL,
    model_version VARCHAR(10) NOT NULL,
    prediction VARCHAR(100) NOT NULL,
    confidence REAL NOT NULL,
    latency REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (input_id) REFERENCES users_inputs(id) ON DELETE CASCADE
);

-- Indexes untuk optimasi query performance
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_users_inputs_consent ON users_inputs(user_consent);
CREATE INDEX IF NOT EXISTS idx_users_inputs_timestamp ON users_inputs(timestamp);

-- Comments untuk dokumentasi
COMMENT ON TABLE users_inputs IS 'Menyimpan input teks dari pengguna dengan consent tracking';
COMMENT ON TABLE predictions IS 'Menyimpan hasil prediksi model dengan metrics';
COMMENT ON COLUMN users_inputs.user_consent IS 'Apakah user mengizinkan data disimpan untuk retraining';
COMMENT ON COLUMN users_inputs.anonymized IS 'Apakah data sudah dianonimkan (PII removed)';
COMMENT ON COLUMN predictions.confidence IS 'Confidence score dari prediksi (0-1)';
COMMENT ON COLUMN predictions.latency IS 'Waktu prediksi dalam seconds';
