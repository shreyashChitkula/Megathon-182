"""
Explainability Module
- Generate visual explanations with heatmaps
- Highlight damaged regions
- Provide textual reasoning
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class ExplainabilityGenerator:
    def __init__(self):
        self.colors = {
            'Bonnet': (255, 0, 0),      # Red
            'Bumper': (0, 255, 0),      # Green
            'Dickey': (0, 0, 255),      # Blue
            'Door': (255, 255, 0),      # Yellow
            'Fender': (255, 0, 255),    # Magenta
            'Light': (0, 255, 255),     # Cyan
            'Windshield': (128, 0, 128) # Purple
        }
    
    def generate_explanation_image(self, original_image_path, yolo_results, detected_parts):
        """
        Generate an annotated image with explanations
        - Bounding boxes with labels
        - Severity indicators
        - Part-wise damage highlights
        - "No Damage Detected" overlay if clean
        """
        # Load image
        img = cv2.imread(original_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Create PIL image for better text rendering
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_large = ImageFont.load_default()
        
        # Check if any damage detected
        has_damage = yolo_results and len(yolo_results[0].boxes) > 0
        
        if has_damage:
            # Draw bounding boxes and labels
            boxes = yolo_results[0].boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                # Get part name
                part_name = self._get_part_name(class_id)
                # Use default color for all parts now (consistent)
                color = (255, 50, 50)  # Red color for damage
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label background
                label = f"{part_name} ({confidence:.2f})"
                bbox = draw.textbbox((x1, y1-25), label, font=font_small)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1-25), label, fill=(255, 255, 255), font=font_small)
        else:
            # No damage detected - show clean status overlay
            # Add semi-transparent green overlay
            overlay = Image.new('RGBA', pil_img.size, (50, 205, 50, 60))  # Green with transparency
            pil_img = pil_img.convert('RGBA')
            pil_img = Image.alpha_composite(pil_img, overlay)
            pil_img = pil_img.convert('RGB')
            draw = ImageDraw.Draw(pil_img)
            
            # Draw "No Damage Detected" banner
            banner_text = "✓ NO DAMAGE DETECTED"
            
            # Calculate text position (center)
            bbox = draw.textbbox((0, 0), banner_text, font=font_large)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = (w - text_width) // 2
            text_y = h // 2 - text_height
            
            # Draw background rectangle for text
            padding = 20
            draw.rectangle(
                [text_x - padding, text_y - padding, 
                 text_x + text_width + padding, text_y + text_height + padding],
                fill=(34, 139, 34),  # Dark green
                outline=(255, 255, 255),
                width=3
            )
            
            # Draw text
            draw.text((text_x, text_y), banner_text, fill=(255, 255, 255), font=font_large)
            
            # Add subtitle
            subtitle = "Vehicle appears to be in clean condition"
            try:
                font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                font_subtitle = font_small
            
            bbox_sub = draw.textbbox((0, 0), subtitle, font=font_subtitle)
            subtitle_width = bbox_sub[2] - bbox_sub[0]
            subtitle_x = (w - subtitle_width) // 2
            subtitle_y = text_y + text_height + padding + 10
            
            draw.text((subtitle_x, subtitle_y), subtitle, fill=(255, 255, 255), font=font_subtitle)
        
        return np.array(pil_img)
    
    def generate_heatmap(self, original_image_path, yolo_results):
        """
        Generate attention/saliency heatmap showing areas of damage
        Shows green overlay if no damage detected
        """
        # Load image
        img = cv2.imread(original_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Check if any damage detected
        has_damage = yolo_results and len(yolo_results[0].boxes) > 0
        
        if has_damage:
            # Create blank heatmap
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            # Add Gaussian blobs for each detected box
            boxes = yolo_results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                radius = int(max((x2 - x1), (y2 - y1)) / 2)
                
                # Create Gaussian blob
                y, x = np.ogrid[:h, :w]
                mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(np.float32)
                gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (radius/2)**2))
                heatmap += mask * gaussian
            
            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            # Apply colormap (red for damage)
            heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            # Overlay on original image
            overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        else:
            # No damage - create clean/green heatmap
            # Create uniform green overlay
            green_overlay = np.zeros_like(img)
            green_overlay[:, :] = [50, 205, 50]  # Lime green color
            
            # Light overlay to show "clean" status
            overlay = cv2.addWeighted(img, 0.85, green_overlay, 0.15, 0)
            
            # Add "CLEAN" watermark
            cv2.putText(overlay, 'CLEAN', 
                       (w//2 - 100, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       2, (34, 139, 34), 4, cv2.LINE_AA)
        
        return overlay
    
    def generate_textual_explanation(self, analysis_results):
        """
        Generate human-readable explanation of the analysis
        """
        explanation = []
        
        # 1. Damage Detection Summary
        detected_parts = analysis_results.get('detected_parts', {})
        if detected_parts:
            explanation.append("## Damage Detection:")
            for part, data in detected_parts.items():
                count = data.get('count', 0)
                explanation.append(f"  • {count} x {part} damaged")
        
        # 2. Severity Assessment
        severity = analysis_results.get('severity', {})
        explanation.append(f"\n## Severity Assessment:")
        explanation.append(f"  • Level: {severity.get('severity_label', 'Unknown')}")
        explanation.append(f"  • Score: {severity.get('severity_score', 0)}/100")
        if severity.get('has_critical_damage'):
            explanation.append(f"  ⚠ Critical parts damaged: {', '.join(severity.get('critical_parts_damaged', []))}")
        
        # 3. Cost Estimation (Indian Rupees)
        cost_band = analysis_results.get('cost_band', {})
        if cost_band:
            explanation.append(f"\n## Cost Estimation (in ₹):")
            base_cost = cost_band.get('base_cost', 0)
            estimated_cost = cost_band.get('estimated_cost', 0)
            cost_band_name = cost_band.get('band') or cost_band.get('cost_band', 'UNKNOWN')
            risk_level = cost_band.get('risk') or cost_band.get('risk_level', 'Unknown')
            
            explanation.append(f"  • Base Cost: ₹{base_cost:,.0f}")
            explanation.append(f"  • Estimated Total: ₹{estimated_cost:,.0f}")
            explanation.append(f"  • Cost Band: {cost_band_name}")
            explanation.append(f"  • Risk Level: {risk_level}")
        else:
            explanation.append(f"\n## Cost Estimation:")
            explanation.append(f"  ⚠ Cost information not available")
        
        # 4. Text-Image Consistency
        consistency = analysis_results.get('consistency', {})
        explanation.append(f"\n## Description Verification:")
        explanation.append(f"  • Consistency Score: {consistency.get('consistency_score', 0)*100:.1f}%")
        explanation.append(f"  • Verdict: {consistency.get('verdict', 'N/A')}")
        if consistency.get('common_parts'):
            explanation.append(f"  • Matched Parts: {', '.join(consistency.get('common_parts', []))}")
        if consistency.get('unmentioned_detected_parts'):
            explanation.append(f"  ⚠ Parts detected but not mentioned: {', '.join(consistency.get('unmentioned_detected_parts', []))}")
        
        # 5. Fraud Indicators
        fraud = analysis_results.get('fraud', {})
        explanation.append(f"\n## Fraud Analysis:")
        
        duplicate = fraud.get('duplicate', {})
        if duplicate.get('is_duplicate'):
            explanation.append(f"  ✗ ALERT: Duplicate image detected ({duplicate.get('confidence', 0)*100:.1f}% match)")
        else:
            explanation.append(f"  ✓ No duplicate images found")
        
        ai_gen = fraud.get('ai_generated', {})
        if ai_gen.get('is_ai_generated'):
            explanation.append(f"  ✗ WARNING: AI-generated image suspected ({ai_gen.get('confidence', 0)*100:.1f}% confidence)")
        else:
            explanation.append(f"  ✓ Image appears authentic")
        
        # 6. Overall Confidence
        confidence = analysis_results.get('confidence', {})
        explanation.append(f"\n## Claim Confidence Score: {confidence.get('confidence_score', 0):.1f}/100")
        explanation.append(f"  • Verdict: {confidence.get('verdict', 'N/A')}")
        explanation.append(f"  • Risk Level: {confidence.get('risk_level', 'Unknown')}")
        explanation.append(f"  • Recommendation: {confidence.get('recommendation', 'N/A')}")
        
        return "\n".join(explanation)
    
    def _get_part_name(self, class_id):
        """Get part name from class ID (17 classes from fine-tuned model)"""
        class_names = {
            0: 'Bodypanel-Dent',
            1: 'Front-Windscreen',
            2: 'Headlight',
            3: 'Rear-Windscreen',
            4: 'RunningBoard',
            5: 'Sidemirror',
            6: 'Signlight',
            7: 'Taillight',
            8: 'Bonnet-Dent',
            9: 'Boot-Dent',
            10: 'Door-Dent',
            11: 'Fender-Dent',
            12: 'Front-Bumper',
            13: 'Pillar-Dent',
            14: 'QuaterPanel',
            15: 'Rear-Bumper',
            16: 'Roof-Dent'
        }
        return class_names.get(int(class_id), 'Unknown')
    
    def save_explanation_report(self, output_path, analysis_results, explanation_img, heatmap_img):
        """
        Save a comprehensive visual report
        """
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot annotated image
        axes[0].imshow(explanation_img)
        axes[0].set_title('Detected Damage (with annotations)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Plot heatmap
        axes[1].imshow(heatmap_img)
        axes[1].set_title('Damage Attention Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
