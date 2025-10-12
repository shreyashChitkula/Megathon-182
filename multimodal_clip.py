"""
Zero-Shot CLIP Multimodal Analyzer
Uses OpenAI's CLIP model for text-image consistency verification
NO FINE-TUNING REQUIRED - Works out of the box!
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class CLIPMultimodalAnalyzer:
    """
    Advanced multimodal analyzer using CLIP (Contrastive Language-Image Pre-training)
    
    Key Features:
    - Zero-shot capable (no fine-tuning needed)
    - Semantic understanding of text-image relationships
    - Handles paraphrasing and synonyms
    - Fast inference (~50ms on CPU)
    
    How it works:
    1. CLIP encodes both image and text into a shared embedding space
    2. Computes cosine similarity between embeddings
    3. Higher similarity = better text-image consistency
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model
        
        Args:
            model_name: HuggingFace model identifier
                Options:
                - "openai/clip-vit-base-patch32" (default, 151M params, faster)
                - "openai/clip-vit-base-patch16" (151M params, better quality)
                - "openai/clip-vit-large-patch14" (428M params, best quality, slower)
        """
        print(f"Loading CLIP model: {model_name}...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print("✓ CLIP model loaded successfully")
        
    def verify_consistency(
        self, 
        image_path: str, 
        description: str, 
        detected_parts: List[str]
    ) -> Dict:
        """
        Verify text-image consistency using CLIP zero-shot classification
        
        Args:
            image_path: Path to the damage image
            description: User's text description of damage
            detected_parts: List of parts detected by YOLO (e.g., ['door', 'bumper'])
            
        Returns:
            Dictionary with:
            - consistency_score: Overall consistency (0-100)
            - description_match: How well description matches image
            - part_scores: Individual scores for each detected part
            - verdict: HIGH/MEDIUM/LOW
            - explanation: Human-readable explanation
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # 1. Check overall description match
            description_score = self._compute_description_similarity(
                image, description
            )
            
            # 2. Check individual part consistency
            part_scores = self._compute_part_similarities(
                image, detected_parts
            )
            
            # 3. Cross-verify: Do detected parts match description?
            cross_verification = self._cross_verify_parts(
                image, description, detected_parts
            )
            
            # 4. Compute weighted consistency score
            consistency_score = self._compute_final_score(
                description_score, 
                part_scores, 
                cross_verification
            )
            
            # 5. Generate verdict and explanation
            verdict = self._get_verdict(consistency_score)
            explanation = self._generate_explanation(
                description_score,
                part_scores,
                cross_verification,
                description,
                detected_parts
            )
            
            return {
                'consistency_score': round(consistency_score * 100, 1),
                'description_match': round(description_score * 100, 1),
                'part_scores': {
                    part: round(score * 100, 1) 
                    for part, score in part_scores.items()
                },
                'cross_verification': round(cross_verification * 100, 1),
                'verdict': verdict,
                'explanation': explanation,
                'matched_parts': [
                    part for part, score in part_scores.items() 
                    if score > 0.6
                ],
                'missing_parts': [
                    part for part, score in part_scores.items() 
                    if score <= 0.6
                ]
            }
            
        except Exception as e:
            print(f"Error in CLIP consistency check: {e}")
            return self._fallback_response()
    
    def _compute_description_similarity(
        self, 
        image: Image.Image, 
        description: str
    ) -> float:
        """
        Compute similarity between image and user description
        """
        # Create prompts
        positive_prompt = f"a photo of a damaged car with {description}"
        negative_prompt = "a photo of an undamaged car in perfect condition"
        
        # Encode
        inputs = self.processor(
            text=[positive_prompt, negative_prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get similarity scores
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
        
        # Return probability of positive match
        return probs[0].item()
    
    def _compute_part_similarities(
        self, 
        image: Image.Image, 
        detected_parts: List[str]
    ) -> Dict[str, float]:
        """
        Compute how well each detected part matches the image
        """
        if not detected_parts:
            return {}
        
        # Create prompts for each part
        prompts = [
            f"a damaged {part}" for part in detected_parts
        ] + ["an undamaged car"]
        
        # Encode
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get similarity scores
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
        
        # Map probabilities to parts
        part_scores = {
            part: probs[i].item() 
            for i, part in enumerate(detected_parts)
        }
        
        return part_scores
    
    def _cross_verify_parts(
        self, 
        image: Image.Image, 
        description: str, 
        detected_parts: List[str]
    ) -> float:
        """
        Verify if detected parts are mentioned in description
        Uses semantic similarity, not just keyword matching
        """
        if not detected_parts:
            return 1.0  # No parts to verify
        
        # Create prompts
        part_list = ", ".join(detected_parts)
        matching_prompt = f"a car with damaged {part_list}, {description}"
        mismatch_prompt = f"a car with different damage than {part_list}"
        
        # Encode
        inputs = self.processor(
            text=[matching_prompt, mismatch_prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get similarity scores
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
        
        return probs[0].item()
    
    def _compute_final_score(
        self, 
        description_score: float,
        part_scores: Dict[str, float],
        cross_verification: float
    ) -> float:
        """
        Weighted combination of all scores
        
        Weights:
        - Description match: 40%
        - Part consistency: 35%
        - Cross-verification: 25%
        """
        # Average part scores
        avg_part_score = (
            np.mean(list(part_scores.values())) 
            if part_scores 
            else 0.5
        )
        
        # Weighted combination
        final_score = (
            0.40 * description_score +
            0.35 * avg_part_score +
            0.25 * cross_verification
        )
        
        return final_score
    
    def _get_verdict(self, score: float) -> str:
        """Convert score to verdict"""
        if score >= 0.75:
            return "HIGH"
        elif score >= 0.50:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_explanation(
        self,
        description_score: float,
        part_scores: Dict[str, float],
        cross_verification: float,
        description: str,
        detected_parts: List[str]
    ) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        # Description match
        if description_score >= 0.75:
            explanations.append(
                f"✓ Description matches image very well ({description_score*100:.1f}%)"
            )
        elif description_score >= 0.50:
            explanations.append(
                f"⚠ Description partially matches image ({description_score*100:.1f}%)"
            )
        else:
            explanations.append(
                f"✗ Description doesn't match image well ({description_score*100:.1f}%)"
            )
        
        # Part consistency
        if part_scores:
            high_confidence_parts = [
                part for part, score in part_scores.items() 
                if score > 0.7
            ]
            low_confidence_parts = [
                part for part, score in part_scores.items() 
                if score <= 0.5
            ]
            
            if high_confidence_parts:
                explanations.append(
                    f"✓ Detected parts confirmed: {', '.join(high_confidence_parts)}"
                )
            if low_confidence_parts:
                explanations.append(
                    f"⚠ Uncertain about: {', '.join(low_confidence_parts)}"
                )
        
        # Cross-verification
        if cross_verification >= 0.70:
            explanations.append(
                "✓ Detected parts align with description"
            )
        elif cross_verification >= 0.50:
            explanations.append(
                "⚠ Some mismatch between detected parts and description"
            )
        else:
            explanations.append(
                "✗ Significant mismatch between detected parts and description"
            )
        
        return " | ".join(explanations)
    
    def _fallback_response(self) -> Dict:
        """Return safe fallback response on error"""
        return {
            'consistency_score': 50.0,
            'description_match': 50.0,
            'part_scores': {},
            'cross_verification': 50.0,
            'verdict': 'MEDIUM',
            'explanation': 'Unable to compute CLIP consistency - using fallback',
            'matched_parts': [],
            'missing_parts': []
        }


# For backward compatibility with existing code
class EnhancedMultimodalAnalyzer(CLIPMultimodalAnalyzer):
    """
    Drop-in replacement for old MultimodalAnalyzer
    Maintains same interface but uses CLIP under the hood
    """
    
    def verify_consistency(
        self, 
        description: str, 
        detected_parts: Dict[str, int],
        image_path: str = None
    ) -> Dict:
        """
        Legacy interface - converts old format to new CLIP format
        
        Args:
            description: User description
            detected_parts: Dict like {'door': 2, 'bumper': 1}
            image_path: Path to image (required for CLIP)
        """
        if image_path is None:
            # Fallback to simple keyword matching
            return self._legacy_keyword_matching(description, detected_parts)
        
        # Convert detected_parts dict to list
        part_list = []
        for part, count in detected_parts.items():
            part_list.extend([part] * count)
        
        # Use CLIP analysis
        result = super().verify_consistency(image_path, description, part_list)
        
        # Convert to legacy format
        return {
            'consistency_score': result['consistency_score'] / 100.0,
            'verdict': result['verdict'],
            'matched_parts': result['matched_parts'],
            'missing_parts': result['missing_parts'],
            'explanation': result['explanation']
        }
    
    def _legacy_keyword_matching(self, description: str, detected_parts: Dict) -> Dict:
        """Fallback to old keyword matching if no image"""
        description_lower = description.lower()
        matched_parts = []
        missing_parts = []
        
        for part in detected_parts.keys():
            if part.lower() in description_lower:
                matched_parts.append(part)
            else:
                missing_parts.append(part)
        
        total_parts = len(detected_parts)
        if total_parts > 0:
            consistency_score = len(matched_parts) / total_parts
        else:
            consistency_score = 1.0
        
        return {
            'consistency_score': consistency_score,
            'matched_parts': matched_parts,
            'missing_parts': missing_parts,
            'verdict': 'HIGH' if consistency_score > 0.7 else 'MEDIUM' if consistency_score > 0.5 else 'LOW',
            'explanation': f"Matched {len(matched_parts)}/{total_parts} parts (keyword-based)"
        }


if __name__ == "__main__":
    # Test the CLIP analyzer
    print("Testing CLIP Multimodal Analyzer...")
    
    analyzer = CLIPMultimodalAnalyzer()
    
    # Example usage
    test_cases = [
        {
            'description': "front bumper is damaged",
            'detected_parts': ['bumper'],
            'expected': 'HIGH'
        },
        {
            'description': "rear door scratched",
            'detected_parts': ['door'],
            'expected': 'HIGH'
        },
        {
            'description': "windshield cracked",
            'detected_parts': ['door', 'bumper'],  # Mismatch
            'expected': 'LOW'
        }
    ]
    
    print("\nTest cases loaded. Ready to analyze real images!")
    print("Usage: analyzer.verify_consistency(image_path, description, detected_parts)")
