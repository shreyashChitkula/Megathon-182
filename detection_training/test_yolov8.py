"""
YOLOv8 Model Testing and Evaluation Script
Tests the trained model on the test set and shows detailed metrics
"""

from ultralytics import YOLO
import os
from pathlib import Path

def test_model(model_path='finetune.pt'):
    """
    Test the trained YOLOv8 model and display comprehensive metrics
    
    Args:
        model_path: Path to the trained model weights
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_yolov8.py")
        return
    
    print(f"{'='*70}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*70}\n")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Run validation on the validation set
    print("Running validation on the validation set...")
    print("-" * 70)
    val_metrics = model.val(data='data.yaml', split='val')
    
    # Display validation metrics
    print(f"\n{'='*70}")
    print("VALIDATION SET METRICS")
    print(f"{'='*70}")
    print(f"mAP50 (IoU=0.50):           {val_metrics.box.map50:.4f}")
    print(f"mAP50-95 (IoU=0.50-0.95):   {val_metrics.box.map:.4f}")
    print(f"mAP75 (IoU=0.75):           {val_metrics.box.map75:.4f}")
    print(f"Precision:                   {val_metrics.box.mp:.4f}")
    print(f"Recall:                      {val_metrics.box.mr:.4f}")
    
    # Per-class metrics
    if hasattr(val_metrics.box, 'maps') and val_metrics.box.maps is not None:
        print(f"\n{'='*70}")
        print("PER-CLASS mAP50 SCORES")
        print(f"{'='*70}")
        
        # Get class names from data.yaml
        class_names = [
            'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
            'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
            'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
            'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
            'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
        ]
        
        for i, (name, map_score) in enumerate(zip(class_names, val_metrics.box.maps)):
            print(f"{i:2d}. {name:30s}: {map_score:.4f}")
    
    # Test on test set if available
    test_images_path = 'test/images'
    if os.path.exists(test_images_path):
        print(f"\n{'='*70}")
        print("TESTING ON TEST SET")
        print(f"{'='*70}")
        
        # Run validation on test set
        test_metrics = model.val(data='data.yaml', split='test')
        
        print(f"mAP50 (IoU=0.50):           {test_metrics.box.map50:.4f}")
        print(f"mAP50-95 (IoU=0.50-0.95):   {test_metrics.box.map:.4f}")
        print(f"mAP75 (IoU=0.75):           {test_metrics.box.map75:.4f}")
        print(f"Precision:                   {test_metrics.box.mp:.4f}")
        print(f"Recall:                      {test_metrics.box.mr:.4f}")
    
    print(f"\n{'='*70}")
    print("GENERATING PREDICTIONS ON TEST IMAGES")
    print(f"{'='*70}")
    
    # Make predictions on test images
    results = model.predict(
        source='test/images',
        save=True,
        conf=0.25,  # confidence threshold
        iou=0.45,   # NMS IoU threshold
        project='runs/detect',
        name='test_predictions',
        exist_ok=True,
        save_txt=True,  # save predictions as txt files
        save_conf=True,  # save confidence scores
    )
    
    print(f"\nPredictions saved to: runs/detect/test_predictions/")
    print(f"Total images processed: {len(results)}")
    
    # Summary statistics
    total_detections = sum([len(r.boxes) for r in results])
    print(f"Total detections made: {total_detections}")
    print(f"Average detections per image: {total_detections/len(results):.2f}")
    
    print(f"\n{'='*70}")
    print("TESTING COMPLETE!")
    print(f"{'='*70}\n")
    
    return val_metrics, test_metrics if os.path.exists(test_images_path) else None


def quick_test_single_image(model_path='runs/detect/car_dent_detection/weights/best.pt', 
                            image_path=None):
    """
    Quick test on a single image
    
    Args:
        model_path: Path to the trained model
        image_path: Path to a test image (if None, uses first test image)
    """
    model = YOLO(model_path)
    
    if image_path is None:
        # Get first image from test set
        test_images = list(Path('test/images').glob('*.*'))
        if test_images:
            image_path = str(test_images[0])
        else:
            print("No test images found!")
            return
    
    print(f"\nTesting on single image: {image_path}")
    results = model.predict(source=image_path, save=True, conf=0.25)
    
    # Print detections
    for r in results:
        boxes = r.boxes
        print(f"\nDetections found: {len(boxes)}")
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            print(f"  Class: {r.names[class_id]}, Confidence: {confidence:.2f}")
    
    return results


if __name__ == '__main__':
    # Run full test
    val_metrics, test_metrics = test_model()
    
    # Uncomment to test on a single image
    # quick_test_single_image()
