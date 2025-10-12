# 🔧 Troubleshooting Guide

## Database Issues

### Phone Number Error (1406)
**Error:** `Data too long for column 'contact_number'`

**Fix:**
```bash
mysql -u root -p < db_migration_phone_fix.sql
```

**Details:** See `DB_PHONE_FIX.md`

---

### Connection Refused
**Error:** `Can't connect to MySQL server`

**Fix:**
```bash
# 1. Check MySQL is running
sudo systemctl status mysql

# 2. Update .env credentials
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
```

---

## Model Issues

### YOLO Model Not Found
**Error:** `FileNotFoundError: yolov8n.pt`

**Fix:**
```bash
# Model auto-downloads on first run
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
mv yolov8n.pt models/
```

---

### Severity Classifier Error
**Error:** `Cannot load resnet50_severity.pth`

**Fix:**
```bash
# Train the model:
python severity_classifier.py --train

# Or use pretrained:
# Download from project repository
```

---

## UI Issues

### Multi-Image Upload Not Working
**Issue:** Only one image uploads

**Fix:** Already fixed in `app.py` - ensure you're using latest version

**Reference:** `MULTI_IMAGE_UPLOAD_FIX.md`

---

### Grad-CAM Not Displaying
**Issue:** Heatmaps don't show

**Fix:** Check `explainability.py` and `gradcam_explainer.py` are properly integrated

**Reference:** `GRADCAM_FEATURE.md`

---

### Cost Calculation Wrong
**Issue:** Incorrect total cost

**Fix:** Verify `car_parts_prices.json` has correct prices

**Reference:** `MULTI_IMAGE_COST_CALCULATION.md`

---

## API Issues

### CLIP Model Timeout
**Error:** `Request timeout`

**Fix:**
```python
# Increase timeout in multimodal_clip.py
requests.post(url, timeout=30)  # Increase from 10 to 30
```

---

### Out of Memory
**Error:** `CUDA out of memory`

**Fix:**
```python
# Reduce batch size or use CPU
device = 'cpu'  # in severity_classifier.py
```

---

## Performance Issues

### Slow Analysis
**Cause:** Running all models sequentially

**Fix:**
- Use GPU if available
- Cache CLIP model
- Optimize image sizes before processing

---

### High Memory Usage
**Cause:** Multiple models loaded simultaneously

**Fix:**
```python
# Implement model unloading after use
del model
torch.cuda.empty_cache()
```

---

## Common Questions

**Q: How to add new car parts?**  
A: Edit `car_parts_prices.json`

**Q: How to change severity thresholds?**  
A: Modify `severity_classifier.py` confidence thresholds

**Q: How to customize fraud detection?**  
A: Update weights in `fraud_detection.py`

**Q: How to export reports?**  
A: Use print CSS or implement PDF export

---

## Still Having Issues?

1. Check console logs: `python app.py`
2. Check browser console: F12 → Console
3. Review recent changes: `RECENT_UPDATES.md`
4. Check system requirements: `README.md`

---

## Report a Bug

Include:
- Error message (full traceback)
- Steps to reproduce
- System info (OS, Python version)
- Screenshot if UI issue
