"""
Claim Confidence Score Module
- Aggregates all analysis factors into a single confidence score
- Provides explainability for the score
"""
import numpy as np


class ClaimConfidenceCalculator:
    def __init__(self):
        # Weights for different factors (must sum to 1.0)
        self.weights = {
            'text_image_consistency': 0.25,
            'authenticity': 0.20,
            'fraud_indicators': 0.30,
            'damage_plausibility': 0.15,
            'claim_frequency': 0.10
        }
    
    def calculate_confidence_score(self, analysis_results):
        """
        Calculate overall claim confidence score (0-100)
        Higher score = More confident the claim is legitimate
        """
        scores = {}
        explanations = []
        
        # 1. Text-Image Consistency (0-100)
        consistency = analysis_results.get('consistency', {})
        consistency_score = consistency.get('consistency_score', 0.5) * 100
        scores['text_image_consistency'] = consistency_score
        
        if consistency_score >= 80:
            explanations.append(f"✓ High consistency between description and detected damage ({consistency_score:.1f}%)")
        elif consistency_score >= 50:
            explanations.append(f"⚠ Moderate consistency ({consistency_score:.1f}%) - Some discrepancies found")
        else:
            explanations.append(f"✗ Low consistency ({consistency_score:.1f}%) - Significant mismatch")
        
        # 2. Claim Authenticity from Text Analysis (0-100)
        authenticity = analysis_results.get('authenticity', {})
        authenticity_score = authenticity.get('authenticity_score', 0.5) * 100
        scores['authenticity'] = authenticity_score
        
        if authenticity_score >= 80:
            explanations.append(f"✓ Description appears authentic ({authenticity_score:.1f}%)")
        else:
            issues = authenticity.get('issues', [])
            explanations.append(f"⚠ Text analysis issues: {', '.join(issues)}")
        
        # 3. Fraud Indicators (0-100, inverted)
        fraud = analysis_results.get('fraud', {})
        
        # Duplicate detection
        duplicate_info = fraud.get('duplicate', {})
        duplicate_penalty = 0
        if duplicate_info.get('is_duplicate'):
            if duplicate_info.get('same_user'):
                duplicate_penalty = 30
                explanations.append("⚠ Similar image submitted before by same user")
            else:
                duplicate_penalty = 60
                explanations.append("✗ ALERT: Similar image found in another user's claim")
        
        # AI-generated detection
        ai_info = fraud.get('ai_generated', {})
        ai_penalty = 0
        if ai_info.get('is_ai_generated'):
            confidence = ai_info.get('confidence', 0) * 100
            ai_penalty = 50
            explanations.append(f"✗ WARNING: Image shows AI-generation patterns ({confidence:.1f}% confidence)")
        
        # Claim frequency
        frequency_info = fraud.get('claim_frequency', {})
        frequency_penalty = 0
        if frequency_info.get('is_suspicious'):
            count = frequency_info.get('claim_count_30d', 0)
            frequency_penalty = min(count * 10, 40)
            explanations.append(f"⚠ User has {count} claims in last 30 days")
        
        fraud_score = max(0, 100 - duplicate_penalty - ai_penalty - frequency_penalty)
        scores['fraud_indicators'] = fraud_score
        
        # 4. Damage Plausibility (0-100)
        severity = analysis_results.get('severity', {})
        severity_level = severity.get('severity', 'MODERATE')
        
        # Plausibility based on severity and cost relationship
        if severity_level == 'MINOR':
            plausibility = 90  # Minor damage is common
        elif severity_level == 'MODERATE':
            plausibility = 85
        else:  # SEVERE
            plausibility = 70  # Severe damage is less common, slightly suspicious
        
        scores['damage_plausibility'] = plausibility
        explanations.append(f"• Damage severity: {severity_level} ({severity.get('severity_score', 0)}/100)")
        
        # 5. Claim Frequency Pattern (already included in fraud)
        scores['claim_frequency'] = 100 - frequency_penalty
        
        # Calculate weighted confidence score
        confidence_score = sum(
            scores[factor] * self.weights[factor]
            for factor in self.weights.keys()
        )
        
        # Generate overall verdict
        verdict = self._get_verdict(confidence_score)
        risk_level = self._get_risk_level(confidence_score)
        
        return {
            'confidence_score': round(confidence_score, 2),
            'verdict': verdict,
            'risk_level': risk_level,
            'component_scores': scores,
            'explanations': explanations,
            'recommendation': self._get_recommendation(confidence_score, fraud)
        }
    
    def _get_verdict(self, score):
        """Get human-readable verdict"""
        if score >= 80:
            return "HIGH CONFIDENCE - Claim appears legitimate"
        elif score >= 60:
            return "MODERATE CONFIDENCE - Minor concerns detected"
        elif score >= 40:
            return "LOW CONFIDENCE - Significant red flags"
        else:
            return "VERY LOW CONFIDENCE - Multiple fraud indicators"
    
    def _get_risk_level(self, score):
        """Get risk level"""
        if score >= 80:
            return "LOW"
        elif score >= 60:
            return "MEDIUM"
        elif score >= 40:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _get_recommendation(self, score, fraud_data):
        """Provide action recommendation"""
        if score >= 80:
            return "APPROVE - Process claim normally"
        elif score >= 60:
            return "REVIEW - Manual verification recommended"
        elif score >= 40:
            return "INVESTIGATE - Detailed investigation required"
        else:
            if fraud_data.get('duplicate', {}).get('is_duplicate'):
                return "REJECT - Duplicate claim detected"
            elif fraud_data.get('ai_generated', {}).get('is_ai_generated'):
                return "REJECT - AI-generated image detected"
            else:
                return "INVESTIGATE - Multiple fraud indicators present"
