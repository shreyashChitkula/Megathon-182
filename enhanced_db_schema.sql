-- Enhanced Database Schema for Insurance Claim Analysis
-- Adds support for fraud detection, claim history, and confidence scoring

USE car_damage_detection;

-- First, add unique index to email if it doesn't exist
ALTER TABLE user_info ADD UNIQUE INDEX idx_email (email);

-- Table to store claim history for fraud detection
CREATE TABLE IF NOT EXISTS claim_history (
    claim_id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(100) NOT NULL,
    image_hash VARCHAR(64),
    claim_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_score FLOAT,
    fraud_flags TEXT,
    claim_description TEXT,
    estimated_cost DECIMAL(10, 2),
    severity_level VARCHAR(20),
    cost_band VARCHAR(20),
    status VARCHAR(20) DEFAULT 'PENDING',
    FOREIGN KEY (email) REFERENCES user_info(email),
    INDEX idx_email_date (email, claim_date),
    INDEX idx_image_hash (image_hash)
);

-- Table to store detailed analysis results
CREATE TABLE IF NOT EXISTS claim_analysis (
    analysis_id INT AUTO_INCREMENT PRIMARY KEY,
    claim_id INT NOT NULL,
    detected_parts JSON,
    text_image_consistency FLOAT,
    authenticity_score FLOAT,
    duplicate_detected BOOLEAN DEFAULT FALSE,
    ai_generated_detected BOOLEAN DEFAULT FALSE,
    damage_severity JSON,
    fraud_indicators JSON,
    confidence_breakdown JSON,
    explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claim_history(claim_id) ON DELETE CASCADE
);

-- Table to track user claim patterns for fraud detection
CREATE TABLE IF NOT EXISTS user_claim_stats (
    user_email VARCHAR(100) PRIMARY KEY,
    total_claims INT DEFAULT 0,
    approved_claims INT DEFAULT 0,
    rejected_claims INT DEFAULT 0,
    pending_claims INT DEFAULT 0,
    average_claim_value DECIMAL(10, 2),
    last_claim_date TIMESTAMP,
    risk_score FLOAT DEFAULT 0.0,
    FOREIGN KEY (user_email) REFERENCES user_info(email)
);

SELECT 'Enhanced database schema created successfully!' as status;
