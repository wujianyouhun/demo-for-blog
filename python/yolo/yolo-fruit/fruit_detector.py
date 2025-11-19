# -*- coding: utf-8 -*-
"""
YOLO Fruit Detection - Optimized Consolidated Version
Supports PyTorch 2.6+ with enhanced performance and English labels
"""
import os
import cv2
import time
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path

warnings.filterwarnings('ignore')

# PyTorch compatibility layer for version 2.6+
import torch
original_torch_load = torch.load

def safe_torch_load(f, *args, **kwargs):
    """Wrapper for torch.load to ensure compatibility with PyTorch 2.6+"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)

torch.load = safe_torch_load

try:
    from ultralytics import YOLO
    import numpy as np
    from collections import defaultdict
    import json
    from datetime import datetime
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Please install required packages: pip install ultralytics opencv-python numpy")
    exit(1)


class FruitDetector:
    """
    Optimized YOLO-based fruit detection system with dual-model approach.
    
    Features:
    - YOLOv8n for fast object detection
    - YOLOv8n-seg for instance segmentation
    - GPU acceleration with automatic fallback to CPU
    - Batch processing with progress tracking
    - English labels for all outputs
    - Performance metrics tracking
    """
    
    # Fruit class mappings (COCO dataset classes)
    FRUIT_CLASSES = {
        'apple': 'Apple',
        'banana': 'Banana',
        'orange': 'Orange',
        'broccoli': 'Broccoli',
        'carrot': 'Carrot',
        'pizza': 'Pizza',
        'cake': 'Cake',
        'sandwich': 'Sandwich',
        'hot dog': 'Hot Dog',
        'donut': 'Donut',
        'cup': 'Cup',
        'fork': 'Fork',
        'knife': 'Knife',
        'spoon': 'Spoon',
        'bowl': 'Bowl',
    }
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        model_size: str = 'n',  # n=nano, s=small, m=medium, l=large, x=xlarge
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the FruitDetector.
        
        Args:
            confidence_threshold: Minimum confidence score for detections (0.0-1.0)
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            verbose: Whether to print detailed progress information
        """
        self.conf_threshold = confidence_threshold
        self.verbose = verbose
        
        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if self.verbose:
            print(f"🚀 Initializing YOLO Fruit Detector")
            print(f"📱 Device: {self.device.upper()}")
            if self.device == 'cuda':
                print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        
        # Setup directories
        self.base_dir = Path(__file__).parent
        self.image_dir = self.base_dir / 'image'
        self.result_dir = self.base_dir / 'result'
        self.result_dir.mkdir(exist_ok=True)
        
        if self.verbose:
            print(f"📂 Image directory: {self.image_dir}")
            print(f"📁 Result directory: {self.result_dir}")
        
        # Load models
        self.model_size = model_size
        self._load_models()
        
        # Performance tracking
        self.processing_times = []
        
    def _load_models(self):
        """Load YOLO detection and segmentation models."""
        try:
            if self.verbose:
                print(f"\n📥 Loading YOLOv8{self.model_size} models...")
            
            start_time = time.time()
            
            # Load models
            detection_model_path = f'yolov8{self.model_size}.pt'
            segmentation_model_path = f'yolov8{self.model_size}-seg.pt'
            
            self.detector_model = YOLO(detection_model_path)
            self.seg_model = YOLO(segmentation_model_path)
            
            load_time = time.time() - start_time
            
            if self.verbose:
                print(f"✅ Models loaded successfully in {load_time:.2f}s")
                
            # Warm up models for consistent performance
            self._warmup_models()
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("💡 Ensure you have internet connection for first-time model download")
            raise
    
    def _warmup_models(self):
        """Warm up models with a dummy inference for consistent performance."""
        if self.verbose:
            print("🔥 Warming up models...")
        
        # Create a small dummy image
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Run inference once to initialize CUDA kernels
        _ = self.detector_model(dummy_img, device=self.device, conf=self.conf_threshold, verbose=False)
        _ = self.seg_model(dummy_img, device=self.device, conf=self.conf_threshold, verbose=False)
        
        if self.verbose:
            print("✅ Warmup complete")
    
    def detect_image(
        self,
        image_path: Path,
        save_results: bool = True
    ) -> Optional[Dict[str, int]]:
        """
        Detect fruits in a single image.
        
        Args:
            image_path: Path to the image file
            save_results: Whether to save visualization results
            
        Returns:
            Dictionary of detected fruits and their counts, or None if error
        """
        if self.verbose:
            print(f"\n🔍 Processing: {image_path.name}")
        
        start_time = time.time()
        
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"❌ Cannot read image: {image_path}")
                return None
            
            # Run detection
            detection_results = self.detector_model(
                img,
                device=self.device,
                conf=self.conf_threshold,
                verbose=False
            )
            
            # Run segmentation
            seg_results = self.seg_model(
                img,
                device=self.device,
                conf=self.conf_threshold,
                verbose=False
            )
            
            # Process results
            fruit_count = defaultdict(int)
            detection_img = img.copy()
            seg_img = img.copy()
            
            # Process detection results
            for result in detection_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id].lower()
                        
                        if class_name in self.FRUIT_CLASSES:
                            # Get box coordinates and confidence
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            
                            # Draw bounding box
                            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add label with English name
                            english_name = self.FRUIT_CLASSES[class_name]
                            label = f"{english_name} {confidence:.2f}"
                            
                            # Calculate text size for background
                            (text_width, text_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            
                            # Draw background rectangle for text
                            cv2.rectangle(
                                detection_img,
                                (x1, y1 - text_height - 10),
                                (x1 + text_width, y1),
                                (0, 255, 0),
                                -1
                            )
                            
                            # Draw text
                            cv2.putText(
                                detection_img,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 0),
                                2
                            )
                            
                            fruit_count[english_name] += 1
            
            # Process segmentation results
            for result in seg_results:
                if result.masks is not None and result.boxes is not None:
                    for i, mask in enumerate(result.masks.data):
                        cls_id = int(result.boxes.cls[i])
                        class_name = result.names[cls_id].lower()
                        
                        if class_name in self.FRUIT_CLASSES:
                            # Process mask
                            mask_np = mask.cpu().numpy()
                            mask_binary = (mask_np * 255).astype(np.uint8)
                            
                            # Ensure 2D mask
                            if len(mask_binary.shape) == 3:
                                mask_binary = mask_binary[0]
                            
                            # Resize mask to image size
                            mask_binary = cv2.resize(
                                mask_binary,
                                (img.shape[1], img.shape[0])
                            )
                            
                            # Create colored mask (yellow)
                            colored_mask = np.zeros_like(img)
                            colored_mask[mask_binary > 0] = [0, 255, 255]
                            
                            # Blend mask with image
                            seg_img = cv2.addWeighted(seg_img, 1, colored_mask, 0.5, 0)
            
            # Save results if requested
            if save_results:
                save_name = image_path.stem
                detection_path = self.result_dir / f"{save_name}_detection.jpg"
                seg_path = self.result_dir / f"{save_name}_segmentation.jpg"
                
                cv2.imwrite(str(detection_path), detection_img)
                cv2.imwrite(str(seg_path), seg_img)
                
                if self.verbose:
                    print(f"💾 Saved: {detection_path.name}")
                    print(f"💾 Saved: {seg_path.name}")
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            if self.verbose:
                fps = 1.0 / processing_time if processing_time > 0 else 0
                print(f"⏱️  Processing time: {processing_time:.3f}s ({fps:.1f} FPS)")
                if fruit_count:
                    print(f"📊 Detected: {dict(fruit_count)}")
                else:
                    print("📊 No fruits detected")
            
            # Clear GPU cache if using CUDA
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return dict(fruit_count) if fruit_count else {}
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            return None
    
    def process_directory(
        self,
        image_dir: Optional[Path] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images (uses self.image_dir if None)
            
        Returns:
            Dictionary mapping image filenames to their detection results
        """
        if image_dir is None:
            image_dir = self.image_dir
        
        print(f"\n{'='*60}")
        print(f"🎯 Starting Batch Processing")
        print(f"{'='*60}")
        
        # Check directory exists
        if not image_dir.exists():
            print(f"❌ Image directory not found: {image_dir}")
            return {}
        
        # Get image files
        supported_formats = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_files = [
            f for f in image_dir.iterdir()
            if f.is_file() and f.suffix in supported_formats
        ]
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return {}
        
        print(f"📸 Found {len(image_files)} images")
        
        # Process each image
        all_results = {}
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}]", end=" ")
            result = self.detect_image(image_file)
            
            if result is not None:
                all_results[image_file.name] = result
        
        # Generate and save report
        self._save_report(all_results)
        
        # Print performance summary
        self._print_performance_summary()
        
        return all_results
    
    def _save_report(self, results: Dict[str, Dict[str, int]]):
        """Save detection report in text and JSON formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Text report
        text_file = self.result_dir / f"detection_report_{timestamp}.txt"
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("🍎 FRUIT DETECTION REPORT 🍎\n")
            f.write("=" * 60 + "\n")
            f.write(f"Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device.upper()}\n")
            f.write(f"Model: YOLOv8{self.model_size}\n")
            f.write(f"Confidence Threshold: {self.conf_threshold}\n")
            f.write(f"Total Images Processed: {len(results)}\n\n")
            
            total_fruits = defaultdict(int)
            
            # Per-image results
            f.write("PER-IMAGE RESULTS:\n")
            f.write("-" * 60 + "\n")
            for image_file, fruits in sorted(results.items()):
                f.write(f"\n📷 {image_file}\n")
                if fruits:
                    for fruit, count in sorted(fruits.items()):
                        f.write(f"   • {fruit}: {count}\n")
                        total_fruits[fruit] += count
                else:
                    f.write("   • No fruits detected\n")
            
            # Summary statistics
            f.write(f"\n{'='*60}\n")
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 60 + "\n")
            
            if total_fruits:
                for fruit, count in sorted(total_fruits.items()):
                    f.write(f"{fruit}: {count}\n")
                f.write(f"\n🎯 Total Fruits Detected: {sum(total_fruits.values())}\n")
            else:
                f.write("No fruits detected in any images\n")
        
        # JSON report
        json_file = self.result_dir / f"detection_report_{timestamp}.json"
        
        report_data = {
            'metadata': {
                'detection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'device': self.device,
                'model': f'YOLOv8{self.model_size}',
                'confidence_threshold': self.conf_threshold,
                'total_images': len(results)
            },
            'results': results,
            'summary': dict(total_fruits),
            'total_fruits': sum(total_fruits.values())
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"📋 Reports saved:")
        print(f"   📄 Text: {text_file.name}")
        print(f"   📄 JSON: {json_file.name}")
    
    def _print_performance_summary(self):
        """Print performance statistics."""
        if not self.processing_times:
            return
        
        avg_time = np.mean(self.processing_times)
        min_time = np.min(self.processing_times)
        max_time = np.max(self.processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"⚡ PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Average Processing Time: {avg_time:.3f}s ({avg_fps:.1f} FPS)")
        print(f"Fastest: {min_time:.3f}s ({1.0/min_time:.1f} FPS)")
        print(f"Slowest: {max_time:.3f}s ({1.0/max_time:.1f} FPS)")
        print(f"Total Images: {len(self.processing_times)}")


def main():
    """Main entry point for the fruit detection system."""
    print("🍎🍌🍊 YOLO FRUIT DETECTION SYSTEM 🍎🍌🍊")
    print("=" * 60)
    print("Optimized Consolidated Version with English Labels")
    print("=" * 60)
    
    try:
        # Initialize detector with optimized settings
        detector = FruitDetector(
            confidence_threshold=0.5,
            model_size='n',  # Use nano model for best speed
            verbose=True
        )
        
        # Process all images
        results = detector.process_directory()
        
        if results:
            print(f"\n{'='*60}")
            print("🎉 DETECTION COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"📁 All results saved to: {detector.result_dir}")
            print(f"📊 Total images processed: {len(results)}")
        else:
            print(f"\n{'='*60}")
            print("⚠️  NO FRUITS DETECTED")
            print(f"{'='*60}")
            print("💡 Tips:")
            print("   • Ensure images are in the 'image' folder")
            print("   • Check image quality and lighting")
            print("   • Try adjusting confidence threshold")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Detection interrupted by user")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR: {e}")
        print(f"{'='*60}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection (needed for first-time model download)")
        print("2. Ensure ultralytics is installed: pip install ultralytics")
        print("3. Verify images exist in the 'image' folder")
        print("4. Check that images are in supported formats (JPG, PNG)")
        
    finally:
        # Restore original torch.load
        torch.load = original_torch_load
        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
