"""
Advanced Multi-GPU Training with Custom DDP Configuration
For maximum control over distributed training across 4 GPUs
"""

from ultralytics import YOLO
import torch
import os
import numpy as np
from ultralytics.utils import callbacks

# Custom callback to print accuracy during training
def on_fit_epoch_end(trainer):
    """Callback to print accuracy metrics after each epoch validation"""
    # Only print on rank 0 in DDP to avoid duplicate outputs
    if hasattr(trainer, 'rank') and trainer.rank != 0:
        return
        
    if trainer.epoch == 0:
        return  # Skip first epoch as metrics may not be fully initialized
    
    epoch = trainer.epoch
    
    # Print epoch accuracy summary
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch + 1}/{trainer.epochs} - ACCURACY REPORT")
    print(f"{'='*80}")
    
    try:
        # Access validation metrics
        if hasattr(trainer, 'validator') and hasattr(trainer.validator, 'metrics'):
            val_metrics = trainer.validator.metrics
            
            # Calculate key accuracy metrics
            precision = val_metrics.box.mp  # Prediction Accuracy: How many predictions were correct
            recall = val_metrics.box.mr     # Detection Accuracy: How many ground truths were found
            map50 = val_metrics.box.map50   # Overall accuracy at IoU=0.5
            map = val_metrics.box.map       # Overall accuracy at IoU=0.5-0.95
            
            # Calculate F1-Score (harmonic mean of precision and recall)
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0
            
            # Print ACCURACY metrics prominently
            print(f"\n┌─ ACCURACY METRICS ─────────────────────────────────────────┐")
            print(f"│                                                             │")
            print(f"│  Overall Label Accuracy (mAP@0.5):     {map50*100:6.2f}%         │")
            print(f"│  Overall Label Accuracy (mAP@0.5-0.95): {map*100:6.2f}%         │")
            print(f"│                                                             │")
            print(f"│  Prediction Accuracy (Precision):      {precision*100:6.2f}%         │")
            print(f"│    → Of all predictions, {precision*100:.1f}% were correct          │")
            print(f"│                                                             │")
            print(f"│  Detection Accuracy (Recall):          {recall*100:6.2f}%         │")
            print(f"│    → Of all ground truths, {recall*100:.1f}% were detected       │")
            print(f"│                                                             │")
            print(f"│  F1-Score (Combined Accuracy):         {f1_score*100:6.2f}%         │")
            print(f"│                                                             │")
            print(f"└─────────────────────────────────────────────────────────────┘")
            
            # Per-class accuracy
            class_names = trainer.data['names']
            per_class_ap50 = val_metrics.box.ap50
            per_class_precision = val_metrics.box.p
            per_class_recall = val_metrics.box.r
            
            if len(per_class_ap50) > 0:
                # Calculate average class accuracy
                avg_class_accuracy = per_class_ap50.mean()
                
                print(f"\n┌─ PER-CLASS ACCURACY ───────────────────────────────────────┐")
                print(f"│ Average Class Accuracy: {avg_class_accuracy*100:.2f}%                          │")
                print(f"├─────────────────────────────────────────────────────────────┤")
                print(f"│ {'Class':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<8}│")
                print(f"├─────────────────────────────────────────────────────────────┤")
                
                for i, class_name in enumerate(class_names.values()):
                    if i < len(per_class_ap50):
                        precision_val = per_class_precision[i] if i < len(per_class_precision) else 0.0
                        recall_val = per_class_recall[i] if i < len(per_class_recall) else 0.0
                        accuracy = per_class_ap50[i]  # mAP50 as accuracy measure
                        print(f"│ {class_name:<30} {accuracy*100:<9.2f}% {precision_val*100:<9.2f}% {recall_val*100:<7.2f}%│")
                
                # Show best and worst performers
                if len(per_class_ap50) > 1:
                    best_idx = per_class_ap50.argmax()
                    worst_idx = per_class_ap50.argmin()
                    print(f"├─────────────────────────────────────────────────────────────┤")
                    print(f"│ Best:  {list(class_names.values())[best_idx]:<30} ({per_class_ap50[best_idx]*100:.2f}%)      │")
                    print(f"│ Worst: {list(class_names.values())[worst_idx]:<30} ({per_class_ap50[worst_idx]*100:.2f}%)      │")
                
                print(f"└─────────────────────────────────────────────────────────────┘")
            
            # Summary of correct vs incorrect predictions (conceptual)
            print(f"\n┌─ LABEL PREDICTION SUMMARY ─────────────────────────────────┐")
            print(f"│                                                             │")
            print(f"│  For every 100 predictions made:                            │")
            print(f"│    • Correct predictions: ~{int(precision*100)}                              │")
            print(f"│    • Incorrect predictions: ~{int((1-precision)*100)}                         │")
            print(f"│                                                             │")
            print(f"│  For every 100 ground truth labels:                        │")
            print(f"│    • Correctly detected: ~{int(recall*100)}                               │")
            print(f"│    • Missed/Not detected: ~{int((1-recall)*100)}                          │")
            print(f"│                                                             │")
            print(f"└─────────────────────────────────────────────────────────────┘")
    
    except Exception as e:
        print(f"│ Note: Accuracy metrics not yet available - {str(e)}")
        print(f"└─────────────────────────────────────────────────────────────┘")
    
    print(f"{'='*80}\n")

def clear_gpu_memory():
    """Clear GPU memory cache on all available GPUs"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n{'='*70}")
        print(f"CLEARING GPU MEMORY")
        print(f"{'='*70}")
        
        for i in range(num_gpus):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if torch.cuda.memory_allocated(i) > 0:
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"  GPU {i}: Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
                else:
                    print(f"  GPU {i}: Memory cleared ✓")
        
        # Final cleanup
        torch.cuda.empty_cache()
        print(f"{'='*70}\n")

def setup_environment():
    """Setup environment variables for optimal multi-GPU performance"""
    # Set NCCL settings for better multi-GPU communication
    os.environ['NCCL_DEBUG'] = 'INFO'  # Set to 'WARN' for less verbose output
    os.environ['NCCL_P2P_DISABLE'] = '0'  # Enable P2P for faster GPU communication
    os.environ['NCCL_IB_DISABLE'] = '0'   # Enable InfiniBand if available
    
    # Set PyTorch distributed settings
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

def train_yolov8_ddp():
    """
    Train YOLOv8 using Distributed Data Parallel across 4 GPUs
    
    Key Settings for 4 GPUs:
    - batch=64 means 16 samples per GPU (64/4)
    - workers=8 means 8 data loading workers per GPU
    - Effective batch size is 64 (4x speedup in training)
    """
    
    setup_environment()
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Verify GPU availability
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*70}")
    print(f"MULTI-GPU TRAINING SETUP")
    print(f"{'='*70}")
    print(f"Total GPUs Available: {num_gpus}")
    
    if num_gpus < 4:
        print(f"\n⚠️  WARNING: Only {num_gpus} GPU(s) detected!")
        print(f"This script is optimized for 4 GPUs but will work with {num_gpus}.")
        response = input(f"Continue with {num_gpus} GPU(s)? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    # Display GPU information
    for i in range(min(num_gpus, 4)):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    print(f"{'='*70}\n")
    
    # Load model - using yolov8m (medium) for better accuracy with multi-GPU
    model = YOLO('yolov8m.pt')
    
    # Add custom callback for epoch-wise accuracy tracking
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    
    print("Starting Distributed Data Parallel Training...")
    print("Accuracy metrics will be printed after each epoch's validation.\n")
    
    # Train with optimized multi-GPU settings
    results = model.train(
        # Data settings
        data='data.yaml',
        
        # Training settings
        epochs=200,
        batch=64,                    # Total batch size (16 per GPU with 4 GPUs)
        imgsz=640,
        
        # Multi-GPU settings
        device=[0, 1, 2, 3],         # Use 4 GPUs
        workers=8,                   # Workers per GPU for data loading
        
        # Output settings
        name='car_dent_detection_4gpu',
        project='runs/detect',
        exist_ok=True,
        save=True,
        save_period=25,              # Save checkpoint every 10 epochs
        
        # Optimization settings
        optimizer='AdamW',           # AdamW often works better for multi-GPU
        lr0=0.04,                   # Learning rate (scaled for batch size)
        lrf=0.16,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,           # Longer warmup for multi-GPU
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Regularization
        dropout=0.0,
        label_smoothing=0.0,
        
        # Augmentation
        hsv_h=0.015,                 # HSV-Hue augmentation
        hsv_s=0.7,                   # HSV-Saturation augmentation
        hsv_v=0.4,                   # HSV-Value augmentation
        degrees=0.0,                 # Rotation augmentation
        translate=0.1,               # Translation augmentation
        scale=0.5,                   # Scale augmentation
        shear=0.0,                   # Shear augmentation
        perspective=0.0,             # Perspective augmentation
        flipud=0.0,                  # Vertical flip probability
        fliplr=0.5,                  # Horizontal flip probability
        mosaic=1.0,                  # Mosaic augmentation probability
        mixup=0.0,                   # MixUp augmentation probability
        copy_paste=0.0,              # Copy-paste augmentation probability
        
        # Performance settings
        amp=True,                    # Automatic Mixed Precision
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=False,         # Set to True for reproducibility (slower)
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        
        # Validation settings
        val=True,
        patience=5,                 # Early stopping patience
        
        # Memory settings
        cache=False,                 # Set to True if you have enough RAM
        fraction=1.0,
        profile=False,
    )
    
    # Get results
    best_model_path = model.trainer.best
    last_model_path = model.trainer.last
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"Best Model:  {best_model_path}")
    print(f"Last Model:  {last_model_path}")
    print(f"Results Dir: runs/detect/car_dent_detection_4gpu/")
    print(f"{'='*70}\n")
    
    # Final validation
    print("Running final validation...")
    metrics = model.val()
    
    print(f"\n{'='*70}")
    print(f"FINAL VALIDATION METRICS - OVERALL PERFORMANCE")
    print(f"{'='*70}")
    print(f"  mAP50:        {metrics.box.map50:.4f}")
    print(f"  mAP50-95:     {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    print(f"  F1-Score:     {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6):.4f}")
    print(f"{'='*70}\n")
    
    # Per-class accuracy metrics
    print(f"{'='*70}")
    print(f"PER-CLASS ACCURACY METRICS")
    print(f"{'='*70}")
    
    class_names = model.names  # Get class names
    per_class_map50 = metrics.box.maps  # Per-class mAP50-95
    per_class_ap50 = metrics.box.ap50  # Per-class AP50
    per_class_precision = metrics.box.p  # Per-class precision
    per_class_recall = metrics.box.r  # Per-class recall
    
    print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    print(f"{'-'*78}")
    
    for i, class_name in enumerate(class_names.values()):
        precision = per_class_precision[i] if i < len(per_class_precision) else 0.0
        recall = per_class_recall[i] if i < len(per_class_recall) else 0.0
        ap50 = per_class_ap50[i] if i < len(per_class_ap50) else 0.0
        map_val = per_class_map50[i] if i < len(per_class_map50) else 0.0
        
        print(f"{class_name:<30} {precision:<12.4f} {recall:<12.4f} {ap50:<12.4f} {map_val:<12.4f}")
    
    print(f"{'='*70}\n")
    
    # Additional statistics
    print(f"{'='*70}")
    print(f"CLASSIFICATION ACCURACY SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Classes:           {len(class_names)}")
    print(f"  Classes with >80% mAP50: {sum(1 for ap in per_class_ap50 if ap > 0.8)}")
    print(f"  Classes with >90% mAP50: {sum(1 for ap in per_class_ap50 if ap > 0.9)}")
    print(f"  Average Class Precision: {per_class_precision.mean():.4f}")
    print(f"  Average Class Recall:    {per_class_recall.mean():.4f}")
    print(f"  Best Performing Class:   {class_names[per_class_ap50.argmax()]} (mAP50: {per_class_ap50.max():.4f})")
    print(f"  Worst Performing Class:  {class_names[per_class_ap50.argmin()]} (mAP50: {per_class_ap50.min():.4f})")
    print(f"{'='*70}\n")
    
    # Clear GPU memory after training
    clear_gpu_memory()
    
    return model, best_model_path


if __name__ == '__main__':
    print("\n🚀 Multi-GPU YOLOv8 Training Script")
    print("This will train using 4 GPUs with Distributed Data Parallel\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA is not available. This script requires GPUs.")
        exit(1)
    
    # Clear GPU memory before starting
    print("Clearing GPU memory before training...")
    clear_gpu_memory()
    
    try:
        # Start training
        trained_model, best_model_path = train_yolov8_ddp()
        
        print("\n✨ Training pipeline completed!")
        print(f"\nTo use the trained model:")
        print(f"  from ultralytics import YOLO")
        print(f"  model = YOLO('{best_model_path}')")
        print(f"  results = model.predict('test/images')\n")
    
    finally:
        # Clear GPU memory after everything completes
        print("\nFinal GPU memory cleanup...")
        clear_gpu_memory()
        print("GPU memory cleared. Training script finished.")
