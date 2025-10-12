"""
Simple YOLOv8 Finetuning Script for Car Dent/Scratch Detection
"""

from ultralytics import YOLO

def train_yolov8():
    """
    Train YOLOv8 model on car dent/scratch detection dataset
    """
    # Load a pretrained YOLOv8 model (yolov8n is the nano/smallest version)
    # You can also use: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt for larger models
    model = YOLO('yolov8n.pt')
    
    # Train the model with memory-efficient settings
    results = model.train(
        data='data.yaml',           # path to dataset configuration
        epochs=50,                  # number of training epochs
        imgsz=640,                   # input image size
        batch=8,                     # batch size (reduced for memory efficiency)
        name='car_dent_detection',   # name of the training run
        patience=4,                 # early stopping patience
        save=True,                   # save checkpoints
        device=0,                    # use GPU 0 (use 'cpu' for CPU training)
        project='runs/detect',       # project directory
        exist_ok=True,               # overwrite existing project
        pretrained=True,             # use pretrained weights
        optimizer='auto',            # optimizer (auto, SGD, Adam, AdamW, etc.)
        verbose=True,                # verbose output
        seed=42,                     # random seed for reproducibility
        single_cls=False,            # treat as single-class dataset
        rect=False,                  # rectangular training
        cos_lr=False,                # cosine learning rate scheduler
        close_mosaic=10,             # disable mosaic augmentation for final epochs
        resume=False,                # resume from last checkpoint
        amp=True,                    # Automatic Mixed Precision training
        fraction=1.0,                # dataset fraction to train on
        profile=False,               # profile ONNX and TensorRT speeds
        freeze=None,                 # freeze layers (None or list of layer indices)
        lr0=0.04,                    # initial learning rate
        lrf=0.1,                    # final learning rate factor
        momentum=0.937,              # SGD momentum/Adam beta1
        weight_decay=0.0005,         # optimizer weight decay
        warmup_epochs=3.0,           # warmup epochs
        warmup_momentum=0.8,         # warmup initial momentum
        warmup_bias_lr=0.1,          # warmup initial bias lr
        workers=4,                   # number of worker threads for data loading (reduced)
        val=True,                    # validate/test during training
        cache=False,                 # don't cache images to save memory
    )
    
    # Get the best model path
    best_model_path = model.trainer.best
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best model saved at: {best_model_path}")
    print(f"{'='*60}\n")
    
    # Validate the model
    print("Validating the best model...")
    metrics = model.val()
    
    print(f"\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    return model, best_model_path


def load_trained_model(model_path='runs/detect/car_dent_detection/weights/best.pt'):
    """
    Load a trained YOLOv8 model
    
    Args:
        model_path: Path to the trained model weights
        
    Returns:
        Loaded YOLO model
    """
    model = YOLO(model_path)
    print(f"Model loaded from: {model_path}")
    return model


if __name__ == '__main__':
    # Train the model
    trained_model, best_model_path = train_yolov8()
    
    # Example: Load the trained model
    # loaded_model = load_trained_model(best_model_path)
    
    # Example: Make predictions with the trained model
    # results = trained_model.predict(source='test/images', save=True, conf=0.25)
