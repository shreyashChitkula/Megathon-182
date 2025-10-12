"""
Detailed Class-wise Accuracy Evaluation for YOLOv8
Compares predicted classes with ground truth labels in the test set
"""

from ultralytics import YOLO
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import json

# Class names from data.yaml
CLASS_NAMES = [
    'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
    'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

def parse_yolo_label(label_file):
    """Parse YOLO format label file"""
    boxes = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height]
                    })
    return boxes

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes (YOLO format)"""
    # Convert from YOLO format (x_center, y_center, w, h) to (x1, y1, x2, y2)
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate_predictions(model_path='runs/detect/car_dent_detection/weights/best.pt',
                        test_images_dir='test/images',
                        test_labels_dir='test/labels',
                        conf_threshold=0.25,
                        iou_threshold=0.5):
    """
    Evaluate model predictions against ground truth labels
    
    Args:
        model_path: Path to trained model
        test_images_dir: Directory containing test images
        test_labels_dir: Directory containing ground truth labels
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching predictions to ground truth
    """
    
    print(f"{'='*70}")
    print(f"CLASS-WISE ACCURACY EVALUATION")
    print(f"{'='*70}\n")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}\n")
    
    # Get all test images
    test_images = list(Path(test_images_dir).glob('*.*'))
    test_images = [img for img in test_images if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"Found {len(test_images)} test images\n")
    
    # Initialize metrics
    class_stats = defaultdict(lambda: {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'total_ground_truth': 0,
        'total_predictions': 0,
        'correct_predictions': 0
    })
    
    total_ground_truth = 0
    total_predictions = 0
    total_correct = 0
    total_matched = 0
    
    # Process each image
    print("Processing test images...")
    for idx, img_path in enumerate(test_images, 1):
        # Get corresponding label file
        label_file = Path(test_labels_dir) / (img_path.stem + '.txt')
        
        # Parse ground truth
        ground_truth = parse_yolo_label(label_file)
        
        # Run prediction
        results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
        
        # Parse predictions
        predictions = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                # Convert xyxy to xywh (normalized)
                xyxy = box.xyxy[0].cpu().numpy()
                img_h, img_w = results[0].orig_shape
                x_center = ((xyxy[0] + xyxy[2]) / 2) / img_w
                y_center = ((xyxy[1] + xyxy[3]) / 2) / img_h
                width = (xyxy[2] - xyxy[0]) / img_w
                height = (xyxy[3] - xyxy[1]) / img_h
                
                predictions.append({
                    'class_id': class_id,
                    'bbox': [x_center, y_center, width, height],
                    'confidence': confidence
                })
        
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        
        # For each prediction, find best matching ground truth
        for pred_idx, pred in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is valid
            if best_iou >= iou_threshold and best_gt_idx != -1:
                matched_pred.add(pred_idx)
                matched_gt.add(best_gt_idx)
                
                pred_class = pred['class_id']
                gt_class = ground_truth[best_gt_idx]['class_id']
                
                # True positive if classes match
                if pred_class == gt_class:
                    class_stats[pred_class]['true_positives'] += 1
                    class_stats[pred_class]['correct_predictions'] += 1
                    total_correct += 1
                else:
                    # Wrong class prediction
                    class_stats[pred_class]['false_positives'] += 1
                    class_stats[gt_class]['false_negatives'] += 1
                
                total_matched += 1
            else:
                # False positive (no good match)
                class_stats[pred['class_id']]['false_positives'] += 1
        
        # Count false negatives (unmatched ground truth)
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx not in matched_gt:
                class_stats[gt['class_id']]['false_negatives'] += 1
        
        # Update totals
        for gt in ground_truth:
            class_stats[gt['class_id']]['total_ground_truth'] += 1
            total_ground_truth += 1
        
        for pred in predictions:
            class_stats[pred['class_id']]['total_predictions'] += 1
            total_predictions += 1
        
        # Progress
        if idx % 20 == 0:
            print(f"Processed {idx}/{len(test_images)} images...")
    
    print(f"Processed all {len(test_images)} images\n")
    
    # Calculate and display metrics
    print(f"{'='*70}")
    print(f"OVERALL METRICS")
    print(f"{'='*70}")
    print(f"Total Ground Truth Boxes:     {total_ground_truth}")
    print(f"Total Predicted Boxes:        {total_predictions}")
    print(f"Correctly Classified Matches: {total_correct}")
    print(f"Total Matched Boxes (IoU≥{iou_threshold}): {total_matched}")
    
    if total_matched > 0:
        classification_accuracy = (total_correct / total_matched) * 100
        print(f"Classification Accuracy:      {classification_accuracy:.2f}%")
    
    if total_ground_truth > 0:
        detection_recall = (total_matched / total_ground_truth) * 100
        print(f"Detection Recall:             {detection_recall:.2f}%")
    
    if total_predictions > 0:
        detection_precision = (total_matched / total_predictions) * 100
        print(f"Detection Precision:          {detection_precision:.2f}%")
    
    # Per-class metrics
    print(f"\n{'='*70}")
    print(f"PER-CLASS METRICS")
    print(f"{'='*70}\n")
    
    print(f"{'Class':<30} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'Acc':>7}")
    print(f"{'-'*70}")
    
    class_metrics = []
    
    for class_id in range(len(CLASS_NAMES)):
        stats = class_stats[class_id]
        class_name = CLASS_NAMES[class_id]
        
        tp = stats['true_positives']
        fp = stats['false_positives']
        fn = stats['false_negatives']
        gt_count = stats['total_ground_truth']
        pred_count = stats['total_predictions']
        
        # Calculate metrics
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        accuracy = (tp / gt_count * 100) if gt_count > 0 else 0
        
        class_metrics.append({
            'class_name': class_name,
            'class_id': class_id,
            'ground_truth': gt_count,
            'predictions': pred_count,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        })
        
        if gt_count > 0 or pred_count > 0:  # Only show classes that appear
            print(f"{class_name:<30} {gt_count:>6} {pred_count:>6} {tp:>6} {fp:>6} {fn:>6} "
                  f"{precision:>6.1f}% {recall:>6.1f}% {accuracy:>6.1f}%")
    
    # Calculate macro averages (average across classes with data)
    active_classes = [m for m in class_metrics if m['ground_truth'] > 0]
    if active_classes:
        avg_precision = np.mean([m['precision'] for m in active_classes])
        avg_recall = np.mean([m['recall'] for m in active_classes])
        avg_accuracy = np.mean([m['accuracy'] for m in active_classes])
        
        print(f"{'-'*70}")
        print(f"{'Macro Average':<30} {'':<6} {'':<6} {'':<6} {'':<6} {'':<6} "
              f"{avg_precision:>6.1f}% {avg_recall:>6.1f}% {avg_accuracy:>6.1f}%")
    
    # Save detailed results to JSON
    results_dict = {
        'model_path': model_path,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'total_images': len(test_images),
        'overall': {
            'total_ground_truth': total_ground_truth,
            'total_predictions': total_predictions,
            'total_correct': total_correct,
            'total_matched': total_matched,
            'classification_accuracy': (total_correct / total_matched * 100) if total_matched > 0 else 0,
            'detection_recall': (total_matched / total_ground_truth * 100) if total_ground_truth > 0 else 0,
            'detection_precision': (total_matched / total_predictions * 100) if total_predictions > 0 else 0,
        },
        'per_class': class_metrics
    }
    
    output_file = 'test_accuracy_report.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Detailed results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    return results_dict


if __name__ == '__main__':
    # Run evaluation
    results = evaluate_predictions(
        model_path='runs/detect/car_dent_detection_4gpu/weights/best.pt',
        test_images_dir='test/images',
        test_labels_dir='test/labels',
        conf_threshold=0.25,
        iou_threshold=0.5
    )
    
    print("\nEvaluation complete!")
    print("\nKey Metrics:")
    print(f"  - Classification Accuracy: How often the predicted class is correct")
    print(f"  - Detection Precision: % of predictions that match ground truth")
    print(f"  - Detection Recall: % of ground truth boxes that were detected")
    print(f"  - Per-class Accuracy: % of ground truth boxes correctly classified per class")
