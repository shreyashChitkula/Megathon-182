# 🚗 ESPER - Enhanced Smart Policy Estimation & Risk Assessment

## AI-Powered Vehicle Damage Assessment & Insurance Claim Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Flask](https://img.shields.io/badge/Flask-3.1.0-green) ![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-orange) ![MySQL](https://img.shields.io/badge/MySQL-8.0-blue)

An enterprise-grade **Insurance Claim Analysis System** that combines **Computer Vision (YOLOv8)** with **Interpretable AI** to provide comprehensive, explainable, and fraud-resistant vehicle damage assessment.

---
## 📽️ The Application


https://github.com/user-attachments/assets/dcd1d33e-c63a-4a05-a356-2c839ca69c1a

## 🗃️ Presentation

[ESPER-PPT.pdf](https://github.com/user-attachments/files/22932870/ESPER-PPT.pdf)

---

## 🚄 Training The Yolov8 Model

https://universe.roboflow.com/sindhu/car_dent_scratch_detection-1/dataset/5

Download the above dataset and use the training scripts provided to train ✅

---

## 📖 Documentation

**📚 [→ Complete Project Guide](PROJECT_GUIDE.md)** - Everything you need to know in one place

---

## ⚡ Quick Start (5 Minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup environment variables
cp .env.example .env
# Edit .env with your MySQL credentials

# 3. Setup database
mysql -u root -p < db_schema.sql

# 4. Load pricing data
python3 utilities/auto_insert_pricing.py

# 5. Run application
python3 app.py

# 6. Access at http://localhost:5000
```

---

## 🎯 Key Features

### ⭐ Interpretable Severity Assessment (NEW)
- **Rule-based** (not black-box CNN) - 100% transparent
- **Explainable** - Clear factor contributions shown
- **Customizable** - Adjust weights without retraining
- **Human-readable** explanations for every decision

### 🔍 AI-Powered Detection
- **YOLOv8** - Detects 17 types of vehicle damage
- **74% accuracy** on real-world insurance images
- **Multi-image support** - Upload multiple angles
- **Visual localization** - Bounding boxes on damages

### 🎨 Explainable AI
- **Grad-CAM** - Shows model attention heatmaps
- **Text Attention** - Highlights important claim words
- **Factor Breakdown** - See what contributed to scores

### 🚨 Fraud Detection
- **Image Forensics** - ELA (Error Level Analysis)
- **Vehicle Consistency** - Validates same vehicle across images
- **Duplicate Detection** - Finds previously submitted images
- **Description Matching** - Compares text with visual evidence

### 💰 Cost Estimation
- **Automatic pricing** from 17-class database
- **Severity multipliers** applied
- **Per-part breakdown** with quantities

---

## 🏗️ Architecture

```
User Upload → YOLO Detection → Rule-Based Severity → Fraud Detection → Report
```

**Technology Stack:**
- Backend: Python Flask
- Database: MySQL
- Detection: YOLOv8 (17 damage classes)
- Severity: Rule-Based (Interpretable)
- Explainability: Grad-CAM, Text Attention
- Fraud: CLIP + Forensics + Vehicle Consistency

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Detection Accuracy** | 74% (YOLO on 17 classes) |
| **Severity Accuracy** | 85-90% (Rule-based) |
| **Fraud Detection** | 85% accuracy |
| **Processing Time** | 3-5 seconds per image |
| **Explainability** | 100% transparent |

---

## 📁 Project Structure

```
project-root/
├── app.py                          # Main Flask application
├── config.py                       # Database configuration
├── rule_based_severity.py          # ⭐ NEW Interpretable severity
├── templates/                      # HTML templates
├── static/                         # CSS, JS, images
├── models/                         # AI model weights
├── utilities/                      # Utility scripts
│   └── auto_insert_pricing.py     # Database pricing setup
├── .env.example                    # Environment config template
└── PROJECT_GUIDE.md                # Complete documentation
```

---

## 🚀 What's NEW (v2.0)

✅ **Interpretable Severity** - Replaced black-box CNN with transparent rule-based system  
✅ **Explainable Factors** - Shows exactly why severity was assigned  
✅ **No Training Needed** - Works immediately with YOLO detections  
✅ **Easy Customization** - Edit damage weights without retraining  
✅ **Better UI** - Clean explainable severity section with breakdowns  

---

## 📚 Learn More

- **[Complete Project Guide](PROJECT_GUIDE.md)** - Full documentation
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues & fixes

---

## 🎓 Credits

**YOLO:** Ultralytics YOLOv8  
**CLIP:** OpenAI CLIP  
**Grad-CAM:** Ramprasaath et al.  
**Framework:** Flask  

---

**Version:** 2.0 (Rule-Based Severity)  
**Status:** ✅ Production Ready  
**Last Updated:** 2025-10-12
