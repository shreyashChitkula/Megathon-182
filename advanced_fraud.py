"""
Advanced Image Forensics Detection
Deep learning-based tampering and AI-generated image detection

Replaces statistical heuristics with CNN-based forensics models

Academic References:
- Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019
- "EfficientNet for Fake Image Detection" (2021) - 94.2% accuracy on CASIA2
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import os
from PIL.ExifTags import TAGS
import io


class ImageForensicsDetector:
    """
    Advanced image forensics detector combining:
    1. Error Level Analysis (ELA)
    2. CNN-based forgery detection
    3. Noise inconsistency analysis
    """
    
    def __init__(self, model_path='models/forensics_model.pth'):
        """
        Initialize forensics detector
        
        Args:
            model_path: Path to pretrained forensics model (e.g., XceptionNet on CASIA2)
        """
        # Use ResNet50 as backbone (can be replaced with XceptionNet)
        self.model = models.wide_resnet50_2(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, 2)  # Real/Fake binary classification
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained forensics weights if available
        if os.path.exists(model_path):
            print(f"Loading forensics model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model_available = True
        else:
            print(f"Warning: {model_path} not found.")
            print("Forensics CNN unavailable. Using ELA + noise analysis only.")
            self.model_available = False
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing for CNN
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def detect_manipulation(self, image_path):
        """
        Comprehensive manipulation detection
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Detailed forensics analysis
        """
        # Step 1: Error Level Analysis
        ela_result = self._compute_ela(image_path)
        
        # Step 2: CNN-based forgery detection (if model available)
        if self.model_available:
            cnn_result = self._cnn_forgery_detection(image_path, ela_result['ela_map'])
            cnn_fake_prob = cnn_result['fake_probability']
        else:
            cnn_fake_prob = 0.5  # Neutral when CNN unavailable
        
        # Step 3: Noise inconsistency analysis
        noise_analysis = self._analyze_noise_inconsistency(image_path)
        
        # Step 4: EXIF metadata check
        exif_result = self._check_exif_metadata(image_path)
        
        # Combined manipulation score
        if self.model_available:
            # Weight CNN heavily when available
            manipulation_score = (
                0.50 * cnn_fake_prob +
                0.25 * ela_result['ela_score'] +
                0.15 * noise_analysis['inconsistency_score'] +
                0.10 * exif_result['suspicion_score']
            )
        else:
            # Fall back to traditional methods
            manipulation_score = (
                0.50 * ela_result['ela_score'] +
                0.30 * noise_analysis['inconsistency_score'] +
                0.20 * exif_result['suspicion_score']
            )
        
        # Determine verdict
        is_manipulated = manipulation_score > 0.65
        
        return {
            'is_manipulated': is_manipulated,
            'confidence': manipulation_score,
            'ela_score': ela_result['ela_score'],
            'cnn_fake_probability': cnn_fake_prob if self.model_available else None,
            'noise_inconsistency': noise_analysis['inconsistency_score'],
            'exif_suspicion': exif_result['suspicion_score'],
            'ela_visualization_path': ela_result.get('visualization_path'),
            'method': 'CNN + ELA + Noise' if self.model_available else 'ELA + Noise',
            'details': {
                'ela': ela_result,
                'noise': noise_analysis,
                'exif': exif_result
            }
        }
    
    def _compute_ela(self, image_path, quality=90, save_visualization=True):
        """
        Error Level Analysis - detects JPEG compression inconsistencies
        
        Principle: Manipulated regions show different compression artifacts
        than original regions because they've been compressed fewer times.
        
        Args:
            image_path: Path to image
            quality: JPEG quality for recompression (default 90)
            save_visualization: Whether to save ELA visualization
            
        Returns:
            dict: ELA analysis results
        """
        try:
            # Load original image
            img = Image.open(image_path).convert('RGB')
            
            # Resave at specified quality
            buffer = io.BytesIO()
            img.save(buffer, 'JPEG', quality=quality)
            buffer.seek(0)
            compressed = Image.open(buffer)
            
            # Compute pixel-wise difference
            ela = ImageChops.difference(img, compressed)
            
            # Enhance differences for visualization
            extrema = ela.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            
            if max_diff == 0:
                scale = 1
                ela_score = 0.0  # No difference = likely authentic
            else:
                scale = 255.0 / max_diff
                ela_enhanced = ImageEnhance.Brightness(ela).enhance(scale)
                ela_array = np.array(ela_enhanced)
                
                # Calculate ELA score (higher = more suspicious)
                # Manipulated images typically have high variance in ELA
                ela_score = min(np.std(ela_array) / 50.0, 1.0)  # Normalize to [0, 1]
            
            # Save visualization
            visualization_path = None
            if save_visualization and max_diff > 0:
                vis_dir = 'static/forensics'
                os.makedirs(vis_dir, exist_ok=True)
                
                filename = os.path.basename(image_path)
                vis_filename = f"ela_{filename}"
                visualization_path = os.path.join(vis_dir, vis_filename)
                
                ela_enhanced.save(visualization_path)
            
            return {
                'ela_map': np.array(ela),
                'ela_score': ela_score,
                'max_difference': max_diff,
                'visualization_path': visualization_path
            }
        
        except Exception as e:
            print(f"ELA computation failed: {e}")
            return {
                'ela_map': None,
                'ela_score': 0.5,  # Neutral score on error
                'max_difference': 0,
                'visualization_path': None
            }
    
    def _cnn_forgery_detection(self, image_path, ela_map):
        """
        CNN-based forgery detection using pretrained model
        
        Args:
            image_path: Path to original image
            ela_map: ELA analysis result (can be None)
            
        Returns:
            dict: CNN prediction results
        """
        try:
            # Load image (prefer ELA map if available for better detection)
            if ela_map is not None and ela_map.any():
                img = Image.fromarray(ela_map.astype('uint8'))
            else:
                img = Image.open(image_path).convert('RGB')
            
            # Preprocess
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                fake_prob = probabilities[1].item()  # Probability of being fake
            
            return {
                'fake_probability': fake_prob,
                'real_probability': probabilities[0].item(),
                'prediction': 'FAKE' if fake_prob > 0.5 else 'REAL'
            }
        
        except Exception as e:
            print(f"CNN forgery detection failed: {e}")
            return {
                'fake_probability': 0.5,
                'real_probability': 0.5,
                'prediction': 'UNKNOWN'
            }
    
    def _analyze_noise_inconsistency(self, image_path, block_size=64):
        """
        Median Noise Inconsistency Analysis
        
        Principle: Authentic images have consistent noise patterns.
        Spliced/manipulated images show noise inconsistencies.
        
        Args:
            image_path: Path to image
            block_size: Size of blocks for local noise estimation
            
        Returns:
            dict: Noise analysis results
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            
            noise_levels = []
            
            # Divide image into blocks and estimate noise per block
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = img[i:i+block_size, j:j+block_size]
                    
                    # Estimate noise level using Laplacian variance
                    noise = cv2.Laplacian(block, cv2.CV_64F).var()
                    noise_levels.append(noise)
            
            if len(noise_levels) == 0:
                return {
                    'inconsistency_score': 0.5,
                    'noise_blocks': 0,
                    'noise_std': 0,
                    'noise_mean': 0
                }
            
            # Calculate inconsistency
            noise_std = np.std(noise_levels)
            noise_mean = np.mean(noise_levels)
            
            # Coefficient of variation as inconsistency measure
            inconsistency_score = min(noise_std / (noise_mean + 1e-6), 1.0)
            
            return {
                'inconsistency_score': inconsistency_score,
                'noise_blocks': len(noise_levels),
                'noise_std': noise_std,
                'noise_mean': noise_mean
            }
        
        except Exception as e:
            print(f"Noise analysis failed: {e}")
            return {
                'inconsistency_score': 0.5,
                'noise_blocks': 0,
                'noise_std': 0,
                'noise_mean': 0
            }
    
    def _check_exif_metadata(self, image_path):
        """
        Check EXIF metadata for signs of manipulation
        
        Args:
            image_path: Path to image
            
        Returns:
            dict: EXIF analysis results
        """
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            
            suspicion_score = 0.0
            flags = []
            
            if exif_data is None:
                # No EXIF data - moderately suspicious
                suspicion_score = 0.6
                flags.append("No EXIF metadata")
            else:
                # Check for camera information
                has_camera_make = False
                has_camera_model = False
                has_software = False
                
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    if tag == 'Make':
                        has_camera_make = True
                    elif tag == 'Model':
                        has_camera_model = True
                    elif tag == 'Software':
                        has_software = True
                        # Check for editing software
                        if any(editor in str(value).lower() for editor in 
                               ['photoshop', 'gimp', 'paint', 'editor']):
                            suspicion_score += 0.3
                            flags.append(f"Edited with: {value}")
                
                # Missing camera info is suspicious
                if not has_camera_make or not has_camera_model:
                    suspicion_score += 0.3
                    flags.append("Missing camera information")
            
            return {
                'suspicion_score': min(suspicion_score, 1.0),
                'flags': flags,
                'has_exif': exif_data is not None
            }
        
        except Exception as e:
            return {
                'suspicion_score': 0.5,
                'flags': ['EXIF check failed'],
                'has_exif': False
            }


class HybridFraudDetector:
    """
    Hybrid fraud detection combining advanced forensics with existing hashing
    """
    def __init__(self, use_forensics=True):
        self.use_forensics = use_forensics
        
        if use_forensics:
            try:
                self.forensics_detector = ImageForensicsDetector()
                print("Advanced forensics detector loaded")
            except Exception as e:
                print(f"Failed to load forensics detector: {e}")
                self.forensics_detector = None
        else:
            self.forensics_detector = None
    
    def comprehensive_fraud_check(self, image_path):
        """
        Comprehensive fraud detection combining multiple methods
        
        Args:
            image_path: Path to image
            
        Returns:
            dict: Complete fraud analysis
        """
        results = {
            'is_fraudulent': False,
            'fraud_score': 0.0,
            'checks': {}
        }
        
        # Advanced forensics check
        if self.forensics_detector is not None:
            forensics_result = self.forensics_detector.detect_manipulation(image_path)
            results['checks']['forensics'] = forensics_result
            results['fraud_score'] = max(results['fraud_score'], forensics_result['confidence'])
            results['is_fraudulent'] = forensics_result['is_manipulated']
        
        # Add other checks (duplicate, AI-generated) here
        # ...
        
        return results
