"""
Multimodal Analysis Module
- Text-Image Consistency Check
- Claim Description Analysis using NLP
- Semantic Matching
"""
from transformers import pipeline
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class MultimodalAnalyzer:
    def __init__(self):
        # Initialize sentiment analyzer (for claim authenticity)
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="distilbert-base-uncased-finetuned-sst-2-english")
        except:
            self.sentiment_analyzer = None
            print("Warning: Sentiment analyzer not loaded")
        
        # Part name mappings - Extended for fine-tuned model (17 classes)
        self.part_keywords = {
            # Generic categories (backward compatibility)
            'Bonnet': ['hood', 'bonnet', 'front cover', 'engine cover'],
            'Bumper': ['bumper', 'front bumper', 'rear bumper'],
            'Dickey': ['trunk', 'dickey', 'boot', 'rear storage'],
            'Door': ['door', 'front door', 'rear door', 'side door', 'driver door', 'passenger door'],
            'Fender': ['fender', 'wheel arch', 'quarter panel'],
            'Light': ['headlight', 'taillight', 'light', 'lamp', 'indicator', 'brake light'],
            'Windshield': ['windshield', 'windscreen', 'glass', 'front glass', 'rear glass', 'window'],
            
            # Detailed categories from fine-tuned model
            'Bodypanel-Dent': ['body panel', 'bodypanel', 'side panel'],
            'Front-Windscreen-Damage': ['front windshield', 'front windscreen', 'front glass'],
            'Headlight-Damage': ['headlight', 'head lamp', 'front light'],
            'Rear-windscreen-Damage': ['rear windshield', 'rear windscreen', 'rear glass', 'back glass'],
            'RunningBoard-Dent': ['running board', 'side step', 'rocker panel'],
            'Sidemirror-Damage': ['side mirror', 'wing mirror', 'rearview mirror'],
            'Signlight-Damage': ['signal light', 'turn signal', 'indicator light'],
            'Taillight-Damage': ['tail light', 'rear light', 'brake light'],
            'bonnet-dent': ['bonnet dent', 'hood dent'],
            'boot-dent': ['boot dent', 'trunk dent'],
            'doorouter-dent': ['door dent', 'door damage', 'outer door'],
            'fender-dent': ['fender dent', 'quarter panel dent'],
            'front-bumper-dent': ['front bumper dent', 'front bumper damage'],
            'pillar-dent': ['pillar dent', 'a-pillar', 'b-pillar', 'c-pillar'],
            'quaterpanel-dent': ['quarter panel dent', 'rear quarter'],
            'rear-bumper-dent': ['rear bumper dent', 'back bumper damage'],
            'roof-dent': ['roof dent', 'roof damage', 'top damage']
        }
        
    def extract_parts_from_text(self, description):
        """Extract mentioned car parts from claim description"""
        description_lower = description.lower()
        mentioned_parts = {}
        
        for part, keywords in self.part_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    mentioned_parts[part] = mentioned_parts.get(part, 0) + 1
                    break
        
        return mentioned_parts
    
    def extract_severity_indicators(self, description):
        """Extract severity indicators from text"""
        description_lower = description.lower()
        
        severity_keywords = {
            'minor': ['scratch', 'minor', 'small', 'light', 'slight', 'cosmetic'],
            'moderate': ['dent', 'damage', 'broken', 'cracked', 'damaged'],
            'severe': ['major', 'severe', 'crushed', 'totaled', 'destroyed', 'smashed', 'collision']
        }
        
        severity_scores = {'minor': 0, 'moderate': 0, 'severe': 0}
        
        for severity, keywords in severity_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    severity_scores[severity] += 1
        
        # Determine overall severity
        max_severity = max(severity_scores, key=severity_scores.get)
        confidence = severity_scores[max_severity] / max(sum(severity_scores.values()), 1)
        
        return {
            'severity': max_severity,
            'confidence': confidence,
            'scores': severity_scores
        }
    
    def check_text_image_consistency(self, description, detected_parts):
        """
        Check consistency between claim description and detected damage
        Returns a consistency score and explanation
        """
        # Extract parts mentioned in text
        text_parts = self.extract_parts_from_text(description)
        
        # Compare with detected parts
        detected_part_names = set(detected_parts.keys())
        mentioned_part_names = set(text_parts.keys())
        
        # Calculate overlap
        common_parts = detected_part_names.intersection(mentioned_part_names)
        only_detected = detected_part_names - mentioned_part_names
        only_mentioned = mentioned_part_names - detected_part_names
        
        # Calculate consistency score
        if len(detected_part_names) == 0 and len(mentioned_part_names) == 0:
            consistency_score = 1.0
        elif len(detected_part_names) == 0 or len(mentioned_part_names) == 0:
            consistency_score = 0.0
        else:
            # Jaccard similarity
            consistency_score = len(common_parts) / len(detected_part_names.union(mentioned_part_names))
        
        # Check quantity consistency
        quantity_match = True
        for part in common_parts:
            detected_count = detected_parts.get(part, {}).get('count', 0)
            mentioned_count = text_parts.get(part, 0)
            if abs(detected_count - mentioned_count) > 2:
                quantity_match = False
                consistency_score *= 0.8  # Penalty for quantity mismatch
        
        return {
            'consistency_score': consistency_score,
            'common_parts': list(common_parts),
            'undetected_mentioned_parts': list(only_mentioned),
            'unmentioned_detected_parts': list(only_detected),
            'quantity_match': quantity_match,
            'verdict': self._get_consistency_verdict(consistency_score)
        }
    
    def _get_consistency_verdict(self, score):
        """Get human-readable verdict"""
        if score >= 0.8:
            return "HIGH - Description matches detected damage"
        elif score >= 0.5:
            return "MEDIUM - Partial match, some discrepancies"
        else:
            return "LOW - Significant mismatch between description and image"
    
    def analyze_claim_authenticity(self, description):
        """Analyze text for signs of fraudulent claims"""
        if not description or len(description.strip()) < 10:
            return {
                'authenticity_score': 0.3,
                'issues': ['Description too brief']
            }
        
        issues = []
        authenticity_score = 1.0
        
        # Check 1: Length (too short or suspiciously long)
        word_count = len(description.split())
        if word_count < 10:
            issues.append("Description too brief")
            authenticity_score -= 0.3
        elif word_count > 500:
            issues.append("Unusually detailed description")
            authenticity_score -= 0.1
        
        # Check 2: Excessive repetition
        words = description.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio < 0.3:
            issues.append("Excessive word repetition")
            authenticity_score -= 0.2
        
        # Check 3: Vague language
        vague_phrases = ['somehow', 'not sure', 'maybe', 'i think', 'probably', 'unclear']
        vague_count = sum(1 for phrase in vague_phrases if phrase in description.lower())
        if vague_count > 2:
            issues.append("Vague or uncertain language")
            authenticity_score -= 0.2
        
        # Check 4: Sentiment analysis (if available)
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(description[:512])[0]
                # Genuine claims often have neutral or slightly negative sentiment
                if sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.9:
                    issues.append("Unusually positive tone for accident claim")
                    authenticity_score -= 0.1
            except:
                pass
        
        return {
            'authenticity_score': max(0.0, authenticity_score),
            'issues': issues if issues else ['No issues detected'],
            'word_count': word_count
        }
