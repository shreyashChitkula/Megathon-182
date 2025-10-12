"""
Vehicle Consistency Checker
Verifies that all uploaded images belong to the same vehicle
Uses multiple techniques: feature matching, color analysis, structural similarity
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Tuple
import imagehash
from PIL import Image


class VehicleConsistencyChecker:
    """
    Analyzes multiple images to determine if they belong to the same vehicle.
    Uses various computer vision techniques for comprehensive verification.
    """
    
    def __init__(self):
        """Initialize the consistency checker with required detectors"""
        # ORB detector for feature matching (more robust than SIFT, no patent issues)
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # BFMatcher for feature matching
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Thresholds for different checks
        self.thresholds = {
            'feature_match_min': 15,      # Minimum good feature matches
            'color_similarity_min': 0.65,  # Minimum color histogram correlation
            'perceptual_hash_max': 15,     # Maximum hash distance (lower = more similar)
            'overall_confidence_min': 0.60 # Minimum overall confidence
        }
    
    def check_consistency(self, image_paths: List[str]) -> Dict:
        """
        Check if all images are from the same vehicle
        
        Args:
            image_paths: List of paths to uploaded images
            
        Returns:
            Dictionary with consistency analysis results
        """
        if len(image_paths) < 2:
            return {
                'is_consistent': True,
                'confidence': 1.0,
                'reason': 'Only one image uploaded - no comparison needed',
                'pairwise_scores': [],
                'warnings': []
            }
        
        print(f"\n🚗 Checking vehicle consistency across {len(image_paths)} images...")
        
        # Load all images
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"⚠ Warning: Could not load image {path}")
                continue
            images.append(img)
        
        if len(images) < len(image_paths):
            return {
                'is_consistent': False,
                'confidence': 0.0,
                'reason': 'Failed to load all images',
                'pairwise_scores': [],
                'warnings': ['Some images could not be loaded']
            }
        
        # Perform pairwise comparisons
        pairwise_scores = []
        warnings = []
        
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                print(f"  Comparing Image {i+1} ↔ Image {j+1}...")
                
                score = self._compare_images(images[i], images[j], i, j)
                pairwise_scores.append({
                    'pair': f"Image {i+1} ↔ Image {j+1}",
                    'image_indices': (i, j),
                    **score
                })
                
                # Check for warnings
                if score['overall_similarity'] < self.thresholds['overall_confidence_min']:
                    warnings.append(
                        f"⚠ Low similarity between Image {i+1} and Image {j+1} "
                        f"({score['overall_similarity']:.1%})"
                    )
        
        # Calculate overall consistency
        avg_similarity = np.mean([s['overall_similarity'] for s in pairwise_scores])
        min_similarity = np.min([s['overall_similarity'] for s in pairwise_scores])
        
        # Determine if consistent
        is_consistent = min_similarity >= self.thresholds['overall_confidence_min']
        
        # Generate reason
        if is_consistent:
            reason = f"All images show high similarity (avg: {avg_similarity:.1%})"
        else:
            reason = f"Some images show low similarity (min: {min_similarity:.1%}). Possible different vehicles!"
        
        result = {
            'is_consistent': is_consistent,
            'confidence': avg_similarity,
            'min_similarity': min_similarity,
            'max_similarity': np.max([s['overall_similarity'] for s in pairwise_scores]),
            'reason': reason,
            'pairwise_scores': pairwise_scores,
            'warnings': warnings,
            'num_comparisons': len(pairwise_scores)
        }
        
        # Print summary
        print(f"\n{'✅' if is_consistent else '❌'} Consistency Check: {reason}")
        print(f"  Average Similarity: {avg_similarity:.1%}")
        print(f"  Min Similarity: {min_similarity:.1%}")
        print(f"  Max Similarity: {result['max_similarity']:.1%}")
        
        if warnings:
            print("\n  Warnings:")
            for warning in warnings:
                print(f"    {warning}")
        
        return result
    
    def _compare_images(self, img1: np.ndarray, img2: np.ndarray, 
                       idx1: int, idx2: int) -> Dict:
        """
        Compare two images using multiple techniques
        
        Args:
            img1, img2: Images to compare
            idx1, idx2: Image indices for logging
            
        Returns:
            Dictionary with comparison scores
        """
        scores = {}
        
        # 1. Feature Matching (ORB)
        try:
            feature_score = self._feature_matching(img1, img2)
            scores['feature_matching'] = feature_score
        except Exception as e:
            print(f"    ⚠ Feature matching failed: {e}")
            scores['feature_matching'] = 0.0
        
        # 2. Color Histogram Similarity
        try:
            color_score = self._color_histogram_similarity(img1, img2)
            scores['color_similarity'] = color_score
        except Exception as e:
            print(f"    ⚠ Color similarity failed: {e}")
            scores['color_similarity'] = 0.0
        
        # 3. Perceptual Hashing
        try:
            hash_score = self._perceptual_hash_similarity(img1, img2)
            scores['perceptual_hash'] = hash_score
        except Exception as e:
            print(f"    ⚠ Perceptual hash failed: {e}")
            scores['perceptual_hash'] = 0.0
        
        # 4. Structural Similarity (on resized versions)
        try:
            ssim_score = self._structural_similarity(img1, img2)
            scores['structural_similarity'] = ssim_score
        except Exception as e:
            print(f"    ⚠ Structural similarity failed: {e}")
            scores['structural_similarity'] = 0.0
        
        # Calculate weighted overall similarity
        # Feature matching is most important, followed by color and hash
        weights = {
            'feature_matching': 0.35,
            'color_similarity': 0.30,
            'perceptual_hash': 0.20,
            'structural_similarity': 0.15
        }
        
        overall = sum(scores[k] * weights[k] for k in weights.keys())
        scores['overall_similarity'] = overall
        
        return scores
    
    def _feature_matching(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Match features between two images using ORB
        
        Returns:
            Similarity score (0-1)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0.0
        
        # Match descriptors
        matches = self.bf_matcher.match(des1, des2)
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Get good matches (top 50% with low distance)
        num_good_matches = len([m for m in matches if m.distance < 50])
        
        # Normalize score
        max_possible_matches = min(len(kp1), len(kp2))
        score = min(1.0, num_good_matches / max(50, max_possible_matches * 0.1))
        
        return score
    
    def _color_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compare color histograms of two images
        
        Returns:
            Similarity score (0-1)
        """
        # Calculate histograms for each channel
        hist1 = []
        hist2 = []
        
        for i in range(3):  # BGR channels
            h1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
            h2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
            
            # Normalize
            h1 = cv2.normalize(h1, h1).flatten()
            h2 = cv2.normalize(h2, h2).flatten()
            
            hist1.extend(h1)
            hist2.extend(h2)
        
        hist1 = np.array(hist1)
        hist2 = np.array(hist2)
        
        # Calculate correlation
        correlation = cv2.compareHist(
            hist1.reshape(-1, 1).astype(np.float32),
            hist2.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        
        # Convert from [-1, 1] to [0, 1]
        score = (correlation + 1) / 2
        
        return max(0.0, min(1.0, score))
    
    def _perceptual_hash_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compare perceptual hashes (pHash) of two images
        
        Returns:
            Similarity score (0-1)
        """
        # Convert to PIL Images
        img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        
        # Calculate perceptual hashes
        hash1 = imagehash.phash(img1_pil)
        hash2 = imagehash.phash(img2_pil)
        
        # Calculate Hamming distance
        distance = hash1 - hash2
        
        # Convert distance to similarity (lower distance = higher similarity)
        # Max distance is 64 for pHash, normalize to 0-1
        similarity = 1.0 - (distance / 64.0)
        
        return max(0.0, min(1.0, similarity))
    
    def _structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate structural similarity (SSIM) between two images
        
        Returns:
            Similarity score (0-1)
        """
        # Resize to same size for comparison
        size = (300, 300)
        img1_resized = cv2.resize(img1, size)
        img2_resized = cv2.resize(img2, size)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        score, _ = ssim(gray1, gray2, full=True)
        
        return max(0.0, min(1.0, score))
    
    def generate_consistency_report(self, consistency_result: Dict) -> str:
        """
        Generate a human-readable consistency report
        
        Args:
            consistency_result: Result from check_consistency()
            
        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 60)
        report.append("VEHICLE CONSISTENCY ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Overall result
        status = "✅ CONSISTENT" if consistency_result['is_consistent'] else "❌ INCONSISTENT"
        report.append(f"\nStatus: {status}")
        report.append(f"Overall Confidence: {consistency_result['confidence']:.1%}")
        report.append(f"Reason: {consistency_result['reason']}")
        
        # Pairwise comparisons
        report.append(f"\nPairwise Comparisons ({consistency_result['num_comparisons']}):")
        report.append("-" * 60)
        
        for score in consistency_result['pairwise_scores']:
            report.append(f"\n{score['pair']}:")
            report.append(f"  Overall Similarity: {score['overall_similarity']:.1%}")
            report.append(f"  Feature Matching: {score['feature_matching']:.1%}")
            report.append(f"  Color Similarity: {score['color_similarity']:.1%}")
            report.append(f"  Perceptual Hash: {score['perceptual_hash']:.1%}")
            report.append(f"  Structural Similarity: {score['structural_similarity']:.1%}")
        
        # Warnings
        if consistency_result['warnings']:
            report.append("\nWarnings:")
            report.append("-" * 60)
            for warning in consistency_result['warnings']:
                report.append(f"  {warning}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Convenience function for quick checks
def verify_same_vehicle(image_paths: List[str]) -> bool:
    """
    Quick check if images are from the same vehicle
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        True if images appear to be from same vehicle
    """
    checker = VehicleConsistencyChecker()
    result = checker.check_consistency(image_paths)
    return result['is_consistent']
