"""
Rule-Based Severity Assessment for ESPER
More interpretable than CNN-based classification

Integrates with existing YOLO detection to provide:
- Transparent severity scoring
- Explainable factors
- Customizable weights per damage type
- Detailed reasoning
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


# ============================================================================
# SEVERITY LEVELS
# ============================================================================

class SeverityLevel(IntEnum):
    """Severity levels matching your existing system."""
    MINIMAL = 0    # No damage or very minor
    MINOR = 1      # Small cosmetic damage
    MODERATE = 2   # Noticeable damage requiring repair
    SEVERE = 3     # Major damage, significant repair
    TOTAL_LOSS = 4 # Vehicle may be write-off


# ============================================================================
# DAMAGE WEIGHTS FOR YOUR 17 CLASSES
# ============================================================================

# Based on safety, structural importance, and repair complexity
DAMAGE_CLASS_WEIGHTS = {
    # CRITICAL (0.85-1.0) - Safety/Structural
    'Front-Windscreen-Damage': 0.95,  # Safety critical - visibility
    'Rear-windscreen-Damage': 0.90,   # Safety critical
    'pillar-dent': 0.85,               # Structural integrity
    
    # HIGH (0.70-0.84) - Functional/Legal
    'Headlight-Damage': 0.80,         # Legal requirement, safety
    'Taillight-Damage': 0.75,         # Legal requirement
    'Signlight-Damage': 0.72,         # Functional importance
    'roof-dent': 0.70,                # Structural but less critical
    
    # MEDIUM-HIGH (0.60-0.69) - Structural panels
    'bonnet-dent': 0.68,              # Engine protection
    'boot-dent': 0.65,                # Trunk area
    'doorouter-dent': 0.63,           # Door panels (safety - side impact)
    'fender-dent': 0.62,              # Body panels
    'quaterpanel-dent': 0.62,         # Quarter panel
    'Sidemirror-Damage': 0.60,        # Functional but replaceable
    
    # MEDIUM (0.50-0.59) - Body panels
    'Bodypanel-Dent': 0.55,           # General body damage
    'front-bumper-dent': 0.52,        # Cosmetic, absorbs impact
    'rear-bumper-dent': 0.50,         # Cosmetic
    
    # LOW (0.40-0.49) - Accessories
    'RunningBoard-Dent': 0.45,        # Cosmetic, easy to replace
}


# Default weight for unknown damage types
DEFAULT_WEIGHT = 0.60


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DamageDetection:
    """Single damage detection from YOLO."""
    class_name: str
    confidence: float
    bbox: List[float]  # [x, y, width, height] normalized
    area: float        # Bounding box area (normalized)


@dataclass
class SeverityAssessment:
    """Complete severity assessment result."""
    severity_level: SeverityLevel
    severity_score: float          # 0.0-1.0
    confidence: float              # Overall confidence
    damage_count: int
    total_damage_area: float       # Normalized area
    critical_damage: bool          # Has critical damage?
    factors: Dict[str, float]      # Contributing factors
    explanation: str               # Human-readable explanation
    damage_breakdown: List[Dict]   # Per-damage details


# ============================================================================
# RULE-BASED SEVERITY ASSESSOR
# ============================================================================

class RuleBasedSeverityAssessor:
    """
    Rule-based severity assessment using damage type weights and heuristics.
    
    Much more interpretable than CNN classifier:
    - Clear weight assignments
    - Transparent scoring rules
    - Explainable factors
    - Easy to customize
    """
    
    def __init__(self, 
                 damage_type_weights: Dict[str, float] = None,
                 area_weight: float = 0.3,
                 count_weight: float = 0.2,
                 type_weight: float = 0.5):
        """
        Initialize assessor.
        
        Args:
            damage_type_weights: Custom weights per damage class
            area_weight: How much total damage area affects score (0-1)
            count_weight: How much damage count affects score (0-1)
            type_weight: How much damage type affects score (0-1)
        """
        self.damage_weights = damage_type_weights or DAMAGE_CLASS_WEIGHTS
        self.area_weight = area_weight
        self.count_weight = count_weight
        self.type_weight = type_weight
        
        # Normalize weights to sum to 1.0
        total = area_weight + count_weight + type_weight
        self.area_weight /= total
        self.count_weight /= total
        self.type_weight /= total
    
    def assess(self, detections: List[DamageDetection]) -> SeverityAssessment:
        """
        Assess overall damage severity.
        
        Args:
            detections: List of damage detections from YOLO
            
        Returns:
            SeverityAssessment with detailed results
        """
        
        # Handle no damage case
        if not detections:
            return SeverityAssessment(
                severity_level=SeverityLevel.MINIMAL,
                severity_score=0.0,
                confidence=1.0,
                damage_count=0,
                total_damage_area=0.0,
                critical_damage=False,
                factors={'no_damage': 0.0},
                explanation="No damage detected",
                damage_breakdown=[]
            )
        
        # Calculate individual factors
        area_score = self._calculate_area_score(detections)
        count_score = self._calculate_count_score(detections)
        type_score = self._calculate_type_score(detections)
        
        # Weighted combination
        severity_score = (
            self.area_weight * area_score +
            self.count_weight * count_score +
            self.type_weight * type_score
        )
        
        # Check for critical damage
        critical_damage = self._has_critical_damage(detections)
        if critical_damage:
            # Boost severity for critical damage
            severity_score = min(1.0, severity_score * 1.2)
        
        # Determine severity level
        severity_level = self._score_to_level(severity_score, len(detections), critical_damage)
        
        # Calculate confidence
        confidence = self._calculate_confidence(detections)
        
        # Total damage area
        total_area = sum(d.area for d in detections)
        
        # Build damage breakdown
        breakdown = self._build_breakdown(detections)
        
        # Generate explanation
        explanation = self._generate_explanation(
            severity_level, 
            detections, 
            area_score, 
            count_score, 
            type_score,
            critical_damage
        )
        
        return SeverityAssessment(
            severity_level=severity_level,
            severity_score=severity_score,
            confidence=confidence,
            damage_count=len(detections),
            total_damage_area=total_area,
            critical_damage=critical_damage,
            factors={
                'area_score': area_score,
                'count_score': count_score,
                'type_score': type_score,
                'critical_multiplier': 1.2 if critical_damage else 1.0
            },
            explanation=explanation,
            damage_breakdown=breakdown
        )
    
    # ========================================================================
    # SCORING METHODS
    # ========================================================================
    
    def _calculate_area_score(self, detections: List[DamageDetection]) -> float:
        """Score based on total damage area (0-1)."""
        total_area = sum(d.area for d in detections)
        
        # Sigmoid-like scaling: more damage area = higher score
        # 0.05 area (5%) → 0.3
        # 0.15 area (15%) → 0.6
        # 0.30 area (30%) → 0.85
        score = 1.0 - np.exp(-total_area * 8)
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_count_score(self, detections: List[DamageDetection]) -> float:
        """Score based on number of damages (0-1)."""
        count = len(detections)
        
        # Logarithmic scaling: diminishing returns for more damages
        # 1 damage → 0.25
        # 3 damages → 0.50
        # 5 damages → 0.65
        # 10 damages → 0.85
        score = np.log(count + 1) / np.log(11)  # log(11) ≈ 2.4
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_type_score(self, detections: List[DamageDetection]) -> float:
        """Score based on damage types and their severity weights (0-1)."""
        if not detections:
            return 0.0
        
        # Weighted average of damage type severities
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for det in detections:
            # Get damage weight (default if unknown)
            damage_weight = self.damage_weights.get(det.class_name, DEFAULT_WEIGHT)
            
            # Weight by confidence
            contribution = damage_weight * det.confidence
            weighted_sum += contribution
            weight_sum += det.confidence
        
        score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        return float(np.clip(score, 0.0, 1.0))
    
    def _has_critical_damage(self, detections: List[DamageDetection]) -> bool:
        """Check if any damage is critical (weight >= 0.85)."""
        for det in detections:
            weight = self.damage_weights.get(det.class_name, DEFAULT_WEIGHT)
            if weight >= 0.85:
                return True
        return False
    
    def _score_to_level(self, score: float, count: int, critical: bool) -> SeverityLevel:
        """
        Convert severity score to discrete level.
        
        Rules:
        - Critical damage → at least MODERATE
        - 5+ damages → at least MODERATE
        - Score-based thresholds otherwise
        """
        
        # Critical damage override
        if critical:
            if score > 0.75:
                return SeverityLevel.SEVERE
            else:
                return SeverityLevel.MODERATE
        
        # Many damages override
        if count >= 7:
            return SeverityLevel.SEVERE
        elif count >= 5:
            return max(SeverityLevel.MODERATE, self._score_threshold(score))
        
        # Standard thresholds
        return self._score_threshold(score)
    
    def _score_threshold(self, score: float) -> SeverityLevel:
        """Map score to severity level via thresholds."""
        if score < 0.15:
            return SeverityLevel.MINIMAL
        elif score < 0.35:
            return SeverityLevel.MINOR
        elif score < 0.60:
            return SeverityLevel.MODERATE
        elif score < 0.85:
            return SeverityLevel.SEVERE
        else:
            return SeverityLevel.TOTAL_LOSS
    
    def _calculate_confidence(self, detections: List[DamageDetection]) -> float:
        """Overall confidence based on detection confidences."""
        if not detections:
            return 1.0
        
        confidences = [d.confidence for d in detections]
        
        # Use harmonic mean (penalizes low confidences more)
        harmonic_mean = len(confidences) / sum(1/c for c in confidences)
        return float(harmonic_mean)
    
    # ========================================================================
    # EXPLANATION GENERATION
    # ========================================================================
    
    def _build_breakdown(self, detections: List[DamageDetection]) -> List[Dict]:
        """Build per-damage breakdown."""
        breakdown = []
        
        for det in detections:
            weight = self.damage_weights.get(det.class_name, DEFAULT_WEIGHT)
            breakdown.append({
                'damage_type': det.class_name,
                'confidence': float(det.confidence),
                'area_percent': float(det.area * 100),
                'severity_weight': float(weight),
                'is_critical': weight >= 0.85
            })
        
        # Sort by severity weight (highest first)
        breakdown.sort(key=lambda x: x['severity_weight'], reverse=True)
        return breakdown
    
    def _generate_explanation(self, 
                            level: SeverityLevel,
                            detections: List[DamageDetection],
                            area_score: float,
                            count_score: float,
                            type_score: float,
                            critical: bool) -> str:
        """Generate human-readable explanation."""
        
        count = len(detections)
        total_area = sum(d.area for d in detections) * 100
        
        # Find most severe damage
        most_severe = max(
            detections, 
            key=lambda d: self.damage_weights.get(d.class_name, DEFAULT_WEIGHT)
        )
        most_severe_name = most_severe.class_name
        most_severe_weight = self.damage_weights.get(most_severe_name, DEFAULT_WEIGHT)
        
        # Build explanation
        parts = []
        
        # Damage count
        if count == 1:
            parts.append(f"Single damage detected: {most_severe_name}")
        else:
            parts.append(f"{count} damages detected")
        
        # Most severe damage
        if most_severe_weight >= 0.85:
            parts.append(f"Critical damage found ({most_severe_name})")
        elif most_severe_weight >= 0.70:
            parts.append(f"High-severity damage present ({most_severe_name})")
        
        # Area coverage
        if total_area > 20:
            parts.append(f"Extensive damage area ({total_area:.1f}%)")
        elif total_area > 10:
            parts.append(f"Significant damage area ({total_area:.1f}%)")
        
        # Severity level reasoning
        if level == SeverityLevel.MINIMAL:
            conclusion = "Overall impact is minimal"
        elif level == SeverityLevel.MINOR:
            conclusion = "Minor cosmetic damage, easily repairable"
        elif level == SeverityLevel.MODERATE:
            conclusion = "Moderate damage requiring professional repair"
        elif level == SeverityLevel.SEVERE:
            conclusion = "Severe damage, significant repair costs expected"
        else:
            conclusion = "Catastrophic damage, vehicle may be total loss"
        
        parts.append(conclusion)
        
        return ". ".join(parts) + "."


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def convert_yolo_to_detections(yolo_results, image_width: int, image_height: int) -> List[DamageDetection]:
    """
    Convert YOLO detection results to DamageDetection format.
    
    Args:
        yolo_results: Results from YOLO model.predict()
        image_width: Original image width
        image_height: Original image height
        
    Returns:
        List of DamageDetection objects
    """
    detections = []
    
    if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
        return detections
    
    boxes = yolo_results.boxes
    
    for i in range(len(boxes)):
        # Get box coordinates (xyxy format in pixels)
        xyxy = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        
        # Convert to normalized [x, y, width, height]
        x = float(x1 / image_width)
        y = float(y1 / image_height)
        w = float((x2 - x1) / image_width)
        h = float((y2 - y1) / image_height)
        
        # Calculate area
        area = w * h
        
        # Get class and confidence
        class_id = int(boxes.cls[i].item())
        class_name = yolo_results.names[class_id]
        confidence = float(boxes.conf[i].item())
        
        detections.append(DamageDetection(
            class_name=class_name,
            confidence=confidence,
            bbox=[x, y, w, h],
            area=area
        ))
    
    return detections


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example usage with mock data."""
    
    # Mock detections
    mock_detections = [
        DamageDetection(
            class_name='Front-Windscreen-Damage',
            confidence=0.92,
            bbox=[0.3, 0.2, 0.4, 0.3],
            area=0.12
        ),
        DamageDetection(
            class_name='front-bumper-dent',
            confidence=0.85,
            bbox=[0.4, 0.6, 0.2, 0.15],
            area=0.03
        ),
        DamageDetection(
            class_name='Headlight-Damage',
            confidence=0.88,
            bbox=[0.2, 0.4, 0.1, 0.08],
            area=0.008
        )
    ]
    
    # Create assessor
    assessor = RuleBasedSeverityAssessor()
    
    # Assess severity
    result = assessor.assess(mock_detections)
    
    # Print results
    print("=" * 70)
    print("RULE-BASED SEVERITY ASSESSMENT")
    print("=" * 70)
    print(f"\nSeverity Level: {result.severity_level.name}")
    print(f"Severity Score: {result.severity_score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Damage Count: {result.damage_count}")
    print(f"Total Area: {result.total_damage_area*100:.1f}%")
    print(f"Critical Damage: {result.critical_damage}")
    
    print(f"\nFactors:")
    for factor, value in result.factors.items():
        print(f"  {factor:20s}: {value:.3f}")
    
    print(f"\nExplanation:")
    print(f"  {result.explanation}")
    
    print(f"\nDamage Breakdown:")
    for i, damage in enumerate(result.damage_breakdown, 1):
        print(f"  {i}. {damage['damage_type']}")
        print(f"     Weight: {damage['severity_weight']:.2f} | "
              f"Conf: {damage['confidence']:.2f} | "
              f"Area: {damage['area_percent']:.1f}% | "
              f"Critical: {damage['is_critical']}")
    
    print("=" * 70)
