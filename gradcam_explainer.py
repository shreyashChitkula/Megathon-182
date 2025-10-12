"""
Grad-CAM Explainability Module
Provides true model-based visual explanations using gradient-based localization

Replaces fake Gaussian overlays with actual model activation heatmaps

Academic Reference:
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks 
  via Gradient-based Localization," ICCV 2017
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import os


class GradCAMExplainer:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explanations
    
    Shows which regions of the image the model focused on for its predictions
    """
    def __init__(self, model, target_layers=None):
        """
        Initialize Grad-CAM explainer
        
        Args:
            model: YOLO or ResNet model
            target_layers: List of layers to target (default: auto-detect)
        """
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Auto-detect target layer if not provided
        if target_layers is None:
            target_layers = self._auto_detect_layers()
        
        self.target_layers = target_layers
        
        # Register hooks for gradient capture
        self._register_hooks()
    
    def _auto_detect_layers(self):
        """Auto-detect the best layer for Grad-CAM"""
        # For YOLO models
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
            # YOLOv8 structure: model.model.model[-2] is last conv layer
            return [self.model.model.model[-2]]
        
        # For ResNet models
        elif hasattr(self.model, 'layer4'):
            # ResNet structure: layer4[-1] is last conv block
            return [self.model.layer4[-1]]
        
        # Default: try to find last convolutional layer
        else:
            conv_layers = []
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    conv_layers.append(module)
            
            if conv_layers:
                return [conv_layers[-1]]
            else:
                raise ValueError("Could not auto-detect target layer. Please specify manually.")
    
    def _register_hooks(self):
        """Register forward and backward hooks for gradient capture"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        for layer in self.target_layers:
            layer.register_forward_hook(forward_hook)
            layer.register_full_backward_hook(backward_hook)
    
    def generate_gradcam(self, image_path, target_class=None, resize=(640, 640)):
        """
        Generate Grad-CAM heatmap for an image
        
        Args:
            image_path: Path to input image
            target_class: Target class index (None for highest prediction)
            resize: Output size
        
        Returns:
            numpy.ndarray: Heatmap overlay on original image
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        img_resized = cv2.resize(img_array, resize)
        
        # Convert to tensor
        img_tensor = self._preprocess_image(img_resized)
        img_tensor.requires_grad = True
        
        # Forward pass
        self.model.eval()
        output = self.model(img_tensor)
        
        # Get target class
        if target_class is None:
            # For YOLO, use detection confidence
            if hasattr(output, 'boxes'):
                # Use first detection's class
                if len(output.boxes) > 0:
                    target_class = int(output.boxes[0].cls[0])
                else:
                    target_class = 0
            else:
                # For classification models
                target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        
        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(output)
        
        if isinstance(output, torch.Tensor):
            one_hot[0, target_class] = 1
        else:
            # Handle YOLO output format
            one_hot = torch.ones(1)
        
        # Backpropagate
        if isinstance(output, torch.Tensor):
            output.backward(gradient=one_hot, retain_graph=True)
        else:
            # For complex outputs, use sum
            try:
                loss = output.sum()
                loss.backward()
            except:
                print("Warning: Could not compute gradients. Returning original image.")
                return img_resized
        
        # Generate CAM
        if self.gradients is not None and self.activations is not None:
            # Global average pooling on gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            
            # Weighted combination of activation maps
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            
            # ReLU to keep only positive influences
            cam = F.relu(cam)
            
            # Normalize
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Resize to input size
            cam = F.interpolate(cam, size=resize, mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
        else:
            # Fallback: return original image
            print("Warning: Gradients not captured. Returning original image.")
            return img_resized
        
        # Apply colormap
        heatmap = cm.jet(cam)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Overlay on original image
        img_normalized = img_resized.astype(np.float32) / 255.0
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
        
        return overlay
    
    def _preprocess_image(self, image):
        """Convert image to tensor"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_pil = Image.fromarray(image)
        return transform(img_pil).unsqueeze(0)
    
    def generate_multi_class_gradcam(self, image_path, class_names, save_dir='static/gradcam'):
        """
        Generate Grad-CAM for multiple damage classes
        
        Args:
            image_path: Path to input image
            class_names: List of class names
            save_dir: Directory to save visualizations
            
        Returns:
            dict: Paths to generated heatmaps per class
        """
        os.makedirs(save_dir, exist_ok=True)
        gradcam_paths = {}
        
        for idx, class_name in enumerate(class_names):
            gradcam = self.generate_gradcam(image_path, target_class=idx)
            
            # Save visualization
            filename = f"gradcam_{class_name.lower()}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(gradcam, cv2.COLOR_RGB2BGR))
            
            gradcam_paths[class_name] = filepath
        
        return gradcam_paths


class SimpleGradCAM:
    """
    Simplified Grad-CAM implementation for easier integration
    Works with any PyTorch model
    """
    def __init__(self):
        pass
    
    def generate_heatmap(self, image_path, model, target_layer_name='layer4'):
        """
        Generate simple activation-based heatmap
        
        Args:
            image_path: Path to image
            model: PyTorch model
            target_layer_name: Name of target layer
            
        Returns:
            numpy.ndarray: Heatmap visualization
        """
        # This is a simplified version that doesn't require gradients
        # Useful for quick visualization when full Grad-CAM is not available
        
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get activations from target layer
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hook
        if hasattr(model, target_layer_name):
            layer = getattr(model, target_layer_name)
            layer.register_forward_hook(get_activation(target_layer_name))
        
        # Forward pass
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img_rgb).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            _ = model(img_tensor)
        
        # Get activation map
        if target_layer_name in activations:
            act = activations[target_layer_name]
            
            # Average across channels
            act_avg = torch.mean(act, dim=1, keepdim=True)
            
            # Resize to original image size
            act_resized = F.interpolate(
                act_avg, 
                size=(img.shape[0], img.shape[1]), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Convert to numpy
            heatmap = act_resized.squeeze().cpu().numpy()
            
            # Normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Apply colormap
            heatmap_colored = cm.jet(heatmap)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            # Overlay
            overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
            
            return overlay
        else:
            print(f"Warning: Layer {target_layer_name} not found")
            return img_rgb
