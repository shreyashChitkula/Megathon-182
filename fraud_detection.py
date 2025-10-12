"""
Fraud Detection Module
- Duplicate Image Detection using Perceptual Hashing
- AI-Generated Image Detection
- Historical Claim Analysis
"""
import imagehash
from PIL import Image
import mysql.connector as connector
import config
import os
import numpy as np
from datetime import datetime, timedelta


class FraudDetector:
    def __init__(self):
        self.duplicate_threshold = 5  # Hamming distance threshold for duplicates
        
    def compute_image_hash(self, image_path):
        """Compute perceptual hash of an image"""
        try:
            img = Image.open(image_path)
            # Use average hash for perceptual similarity
            avg_hash = imagehash.average_hash(img, hash_size=16)
            return str(avg_hash)
        except Exception as e:
            print(f"Error computing hash: {e}")
            return None
    
    def check_duplicate_images(self, image_path, user_email):
        """Check if image has been submitted before"""
        current_hash = self.compute_image_hash(image_path)
        if not current_hash:
            return {"is_duplicate": False, "confidence": 0}
        
        connection = connector.connect(**config.mysql_credentials)
        if connection:
            try:
                with connection.cursor(dictionary=True) as cursor:
                    # Get recent claims from this user and others
                    cursor.execute("""
                        SELECT image_hash, claim_date, email 
                        FROM claim_history 
                        WHERE claim_date > %s
                        ORDER BY claim_date DESC
                        LIMIT 100
                    """, (datetime.now() - timedelta(days=90),))
                    
                    results = cursor.fetchall()
                    
                    for record in results:
                        if record['image_hash']:
                            stored_hash = imagehash.hex_to_hash(record['image_hash'])
                            current_hash_obj = imagehash.hex_to_hash(current_hash)
                            distance = current_hash_obj - stored_hash
                            
                            if distance <= self.duplicate_threshold:
                                is_same_user = record['email'] == user_email
                                return {
                                    "is_duplicate": True,
                                    "confidence": 1.0 - (distance / 64.0),
                                    "same_user": is_same_user,
                                    "previous_claim_date": record['claim_date'],
                                    "severity": "HIGH" if not is_same_user else "MEDIUM"
                                }
            except Exception as e:
                print(f"Error checking duplicates: {e}")
            finally:
                connection.close()
        
        return {"is_duplicate": False, "confidence": 0}
    
    def detect_ai_generated(self, image_path):
        """
        Detect AI-generated images using statistical analysis
        - Check for unusual noise patterns
        - Analyze frequency domain characteristics
        - Check metadata
        """
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Check 1: EXIF metadata (AI-generated images often lack camera metadata)
            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            has_camera_data = False
            if exif_data:
                camera_tags = [271, 272, 305]  # Make, Model, Software
                has_camera_data = any(tag in exif_data for tag in camera_tags)
            
            # Check 2: Noise analysis (AI images have different noise characteristics)
            # Calculate local variance to detect unnatural smoothness
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            local_var = np.var(gray)
            noise_score = 1.0 if local_var < 50 else 0.0  # Low variance = suspicious
            
            # Check 3: Check for perfect symmetry (common in AI-generated images)
            left_half = img_array[:, :img_array.shape[1]//2]
            right_half = np.fliplr(img_array[:, img_array.shape[1]//2:])
            if left_half.shape == right_half.shape:
                symmetry_score = 1.0 - np.mean(np.abs(left_half - right_half)) / 255.0
            else:
                symmetry_score = 0.0
            
            # Aggregate scores
            ai_confidence = (
                (0.3 * (0 if has_camera_data else 1)) +
                (0.4 * noise_score) +
                (0.3 * (symmetry_score if symmetry_score > 0.95 else 0))
            )
            
            return {
                "is_ai_generated": ai_confidence > 0.6,
                "confidence": ai_confidence,
                "indicators": {
                    "missing_camera_metadata": not has_camera_data,
                    "unnatural_smoothness": noise_score > 0.5,
                    "high_symmetry": symmetry_score > 0.95
                }
            }
            
        except Exception as e:
            print(f"Error detecting AI-generated image: {e}")
            return {"is_ai_generated": False, "confidence": 0}
    
    def analyze_claim_frequency(self, user_email):
        """Check for suspicious claim patterns"""
        connection = connector.connect(**config.mysql_credentials)
        if connection:
            try:
                with connection.cursor(dictionary=True) as cursor:
                    # Count claims in last 30 days
                    cursor.execute("""
                        SELECT COUNT(*) as claim_count 
                        FROM claim_history 
                        WHERE email = %s AND claim_date > %s
                    """, (user_email, datetime.now() - timedelta(days=30)))
                    
                    result = cursor.fetchone()
                    claim_count = result['claim_count'] if result else 0
                    
                    # Suspicious if more than 3 claims in 30 days
                    is_suspicious = claim_count > 3
                    return {
                        "is_suspicious": is_suspicious,
                        "claim_count_30d": claim_count,
                        "severity": "HIGH" if claim_count > 5 else "MEDIUM" if is_suspicious else "LOW"
                    }
            except Exception as e:
                print(f"Error analyzing claim frequency: {e}")
            finally:
                connection.close()
        
        return {"is_suspicious": False, "claim_count_30d": 0, "severity": "LOW"}
