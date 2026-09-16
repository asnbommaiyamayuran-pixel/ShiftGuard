"""
================================================================================
ShiftGuard: Custom Image Inference & Prediction Evaluation Suite
================================================================================
Demonstrates Option 1: Loads trained checkpoint 'shiftguard_checkpoint.pth' and
runs end-to-end inference on individual custom image files, evaluating confidence,
predictive entropy, shift signatures, selective adapter weights, and TRUSTED vs UNCERTAIN decisions.
================================================================================
"""

import os
import torch
from PIL import Image, ImageDraw

from shiftguard import set_seed, ID_CLASSES
from shiftguard_pipeline import load_checkpoint, predict_image


def generate_sample_test_images(output_dir: str = "./test_images"):
    """Generates sample custom test images to evaluate inference pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    set_seed(100)
    
    test_cases = [
        ("daytime_car.png", "car", (200, 40, 40), (220, 220, 255)),            # Clean Day Car
        ("nighttime_truck.png", "truck", (20, 50, 100), (10, 15, 30)),        # Dark Night Truck (Domain Shift)
        ("noisy_bike.png", "bike", (10, 10, 10), (180, 180, 180)),             # Sensor Noisy Bike (Modality Shift)
        ("novel_ood_object.png", "unknown", (120, 200, 80), (240, 240, 240)),  # Unseen Object (Semantic Shift)
    ]
    
    generated_paths = []
    for filename, cls, color, bg_color in test_cases:
        img = Image.new("RGB", (64, 64), color=bg_color)
        draw = ImageDraw.Draw(img)
        cx, cy = 32, 32
        
        if cls == "car":
            draw.rectangle([cx - 15, cy - 8, cx + 15, cy + 8], fill=color)
            draw.ellipse([cx - 10, cy + 5, cx - 4, cy + 11], fill=(20, 20, 20))
            draw.ellipse([cx + 4, cy + 5, cx + 10, cy + 11], fill=(20, 20, 20))
        elif cls == "truck":
            draw.rectangle([cx - 18, cy - 12, cx + 18, cy + 10], fill=color)
        elif cls == "bike":
            draw.line([cx - 12, cy, cx + 12, cy], fill=color, width=3)
            draw.ellipse([cx - 14, cy - 6, cx - 2, cy + 6], outline=(0, 0, 0), width=2)
            draw.ellipse([cx + 2, cy - 6, cx + 14, cy + 6], outline=(0, 0, 0), width=2)
        else:
            draw.polygon([(cx, cy - 15), (cx + 15, cy + 15), (cx - 15, cy + 15)], fill=color)
            
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        generated_paths.append((filename, filepath))
        
    return generated_paths


def run_custom_inference_demo():
    print("=" * 80)
    print(" SHIFTGUARD CUSTOM IMAGE INFERENCE EVALUATION")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    checkpoint_path = "shiftguard_checkpoint.pth"
    print(f"\n---> Loading trained model weights from '{checkpoint_path}'...")
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: '{checkpoint_path}' not found. Training quick demo checkpoint...")
        from shiftguard_pipeline import demonstrate_pipeline
        demonstrate_pipeline()
        
    model, epoch, loss, acc = load_checkpoint(checkpoint_path, device=device)
    
    print("\n---> Generating custom test image files...")
    test_files = generate_sample_test_images(output_dir="./test_images")
    
    print("\n" + "=" * 80)
    print(" INFERENCE RESULTS ON CUSTOM IMAGES")
    print("=" * 80)
    
    for filename, filepath in test_files:
        result = predict_image(model, filepath, device=device)
        
        print(f"\n[Test Image File]: {filename}")
        print(f"  Predicted Class   : {result['predicted_class']}")
        print(f"  Confidence        : {result['confidence']:.4f}")
        print(f"  Predictive Entropy: {result['entropy']:.4f}")
        print(f"  Decision Outcome  : {result['decision']}")
        print(f"  Shift Signature   : Domain={result['shift_signature']['domain_shift']:.2f}, "
              f"Modality={result['shift_signature']['modality_shift']:.2f}, "
              f"Semantic={result['shift_signature']['semantic_shift']:.2f}")
        print(f"  Adapter Weights   : Domain={result['adapter_weights']['domain_adapter']:.2f}, "
              f"Modality={result['adapter_weights']['modality_adapter']:.2f}, "
              f"Semantic={result['adapter_weights']['semantic_adapter']:.2f}")
        print("-" * 80)
        
    print("=" * 80)


if __name__ == "__main__":
    run_custom_inference_demo()
