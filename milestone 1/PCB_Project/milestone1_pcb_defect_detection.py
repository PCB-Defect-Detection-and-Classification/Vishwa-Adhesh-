"""
MILESTONE 1: PCB Defect Detection - Image Processing Pipeline
Step-by-step implementation: Dataset Preparation & Image Processing
"""

import cv2
import os
import numpy as np
from pathlib import Path

class PCBDefectDetector:
    """
    Simple PCB Defect Detection Pipeline
    Steps: Alignment → Subtraction → Threshold → Noise Removal
    """
    
    def __init__(self, template_path, test_path, output_dir="output"):
        """
        Initialize with template and test image paths
        
        Args:
            template_path: Path to perfect PCB image
            test_path: Path to PCB with defect
            output_dir: Directory to save results
        """
        self.template_path = template_path
        self.test_path = test_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Will store images at each step
        self.template = None
        self.test = None
        self.template_gray = None
        self.test_gray = None
        self.diff = None
        self.thresh = None
        self.clean = None
    
    # ============ STEP 2: IMAGE ALIGNMENT ============
    def load_and_align(self, target_size=(640, 480)):
        """
        STEP 2: Load images and make them same size
        
        Why?
            - Template and test images may have different sizes
            - Computer can only compare images of same dimensions
            - Grayscale conversion simplifies comparison
        
        Process:
            1. Read both images (template = good, test = may have defect)
            2. Resize both to same dimensions
            3. Convert to grayscale (black & white)
        
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("STEP 2: IMAGE ALIGNMENT & CONVERSION")
        print("="*60)
        
        try:
            # Read images in color first
            self.template = cv2.imread(self.template_path)
            self.test = cv2.imread(self.test_path)
            
            if self.template is None:
                print(f"❌ Error: Cannot load template image from {self.template_path}")
                return False
            
            if self.test is None:
                print(f"❌ Error: Cannot load test image from {self.test_path}")
                return False
            
            print(f"✔️ Template image loaded: {self.template.shape}")
            print(f"✔️ Test image loaded: {self.test.shape}")
            
            # Resize both to same size
            self.template = cv2.resize(self.template, target_size)
            self.test = cv2.resize(self.test, target_size)
            
            print(f"✔️ Both images resized to: {target_size}")
            
            # Convert to grayscale (remove color, keep brightness)
            self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            self.test_gray = cv2.cvtColor(self.test, cv2.COLOR_BGR2GRAY)
            
            print(f"✔️ Both images converted to grayscale")
            
            # Save aligned images for inspection
            cv2.imwrite(os.path.join(self.output_dir, "01_template_aligned.png"), 
                       self.template_gray)
            cv2.imwrite(os.path.join(self.output_dir, "01_test_aligned.png"), 
                       self.test_gray)
            
            print(f"✔️ Aligned images saved to {self.output_dir}")
            print("✅ STEP 2 COMPLETE: Images are same size and grayscale\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during alignment: {str(e)}")
            return False
    
    # ============ STEP 3: IMAGE SUBTRACTION ============
    def compute_difference(self):
        """
        STEP 3: Find differences between template and test
        
        Why?
            - Same parts of both images should look identical
            - When we subtract perfect image from test image:
              * Same regions → become black (0 difference)
              * Different regions (defects) → become bright (high difference)
        
        Process:
            Use absolute difference: |test_gray - template_gray|
        
        Result:
            - Black areas = no defect
            - Bright/white areas = DEFECT LOCATIONS!
        """
        print("\n" + "="*60)
        print("STEP 3: IMAGE SUBTRACTION (MOST IMPORTANT!)")
        print("="*60)
        
        if self.template_gray is None or self.test_gray is None:
            print("❌ Error: Images not aligned yet. Run load_and_align() first!")
            return False
        
        try:
            # Compute absolute difference
            self.diff = cv2.absdiff(self.template_gray, self.test_gray)
            
            print(f"✔️ Difference image computed")
            print(f"   Min pixel value: {self.diff.min()}")
            print(f"   Max pixel value: {self.diff.max()}")
            print(f"   Mean pixel value: {self.diff.mean():.2f}")
            
            # Save difference image
            cv2.imwrite(os.path.join(self.output_dir, "02_difference_raw.png"), 
                       self.diff)
            
            print(f"✔️ Difference image saved to {self.output_dir}")
            print("✅ STEP 3 COMPLETE: Difference image shows potential defects\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during subtraction: {str(e)}")
            return False
    
    # ============ STEP 4: THRESHOLDING ============
    def threshold_defects(self):
        """
        STEP 4: Convert gray difference image to pure black & white
        
        Why?
            - Difference image is gray (many shades)
            - Computer needs CLEAR decision: defect or not?
            - We use Otsu's method: automatically find best split point
        
        Process:
            - Otsu's algorithm finds optimal threshold value automatically
            - Pixels > threshold = WHITE (255) = DEFECT
            - Pixels ≤ threshold = BLACK (0) = OK
        
        Result:
            - Clear black & white image
            - Defect areas clearly visible as white
        """
        print("\n" + "="*60)
        print("STEP 4: THRESHOLD WITH OTSU METHOD")
        print("="*60)
        
        if self.diff is None:
            print("❌ Error: Difference image not computed yet!")
            return False
        
        try:
            # Otsu's thresholding: automatically finds best threshold
            otsu_threshold, self.thresh = cv2.threshold(
                self.diff, 
                0, 
                255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            print(f"✔️ Otsu threshold automatically set to: {otsu_threshold}")
            
            # Count defect pixels
            defect_pixels = np.count_nonzero(self.thresh)
            total_pixels = self.thresh.shape[0] * self.thresh.shape[1]
            defect_percentage = (defect_pixels / total_pixels) * 100
            
            print(f"✔️ Defect pixels found: {defect_pixels} ({defect_percentage:.2f}%)")
            
            # Save threshold image
            cv2.imwrite(os.path.join(self.output_dir, "03_threshold_otsu.png"), 
                       self.thresh)
            
            print(f"✔️ Threshold image saved to {self.output_dir}")
            print("✅ STEP 4 COMPLETE: Clear black & white defect image\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during thresholding: {str(e)}")
            return False
    
    # ============ STEP 5: NOISE REMOVAL ============
    def remove_noise(self, kernel_size=(3, 3)):
        """
        STEP 5: Clean up small noise, keep real defects
        
        Why?
            - Thresholded image may have small noise (random dots)
            - Real defects are usually bigger and solid
            - We use morphological operations:
              * Erosion: removes small white dots
              * Dilation: makes remaining defects clearer and bigger
              * Opening = Erosion then Dilation: removes noise
        
        Process:
            1. Create a kernel (small filter)
            2. Apply MORPH_OPEN: erode then dilate
        
        Result:
            - Noise removed
            - Defects remain clean and solid
        """
        print("\n" + "="*60)
        print("STEP 5: NOISE REMOVAL (MORPHOLOGICAL OPERATIONS)")
        print("="*60)
        
        if self.thresh is None:
            print("❌ Error: Threshold image not computed yet!")
            return False
        
        try:
            # Create structuring element (kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            
            # Apply morphological opening (erosion + dilation)
            # This removes small noise while keeping larger defects
            self.clean = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel)
            
            print(f"✔️ Morphological opening applied (kernel size: {kernel_size})")
            
            # Count remaining defect pixels after cleaning
            clean_defect_pixels = np.count_nonzero(self.clean)
            original_defect_pixels = np.count_nonzero(self.thresh)
            
            print(f"✔️ Defect pixels before cleaning: {original_defect_pixels}")
            print(f"✔️ Defect pixels after cleaning: {clean_defect_pixels}")
            print(f"✔️ Noise removed: {original_defect_pixels - clean_defect_pixels} pixels")
            
            # Save clean image
            cv2.imwrite(os.path.join(self.output_dir, "04_noise_removed_final.png"), 
                       self.clean)
            
            print(f"✔️ Clean defect mask saved to {self.output_dir}")
            print("✅ STEP 5 COMPLETE: Final clean defect mask ready\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during noise removal: {str(e)}")
            return False
    
    # ============ MAIN PIPELINE ============
    def run_pipeline(self, target_size=(640, 480)):
        """
        Run complete pipeline: Steps 2, 3, 4, 5
        """
        print("\n" + "🔵"*30)
        print("MILESTONE 1: PCB DEFECT DETECTION PIPELINE")
        print("🔵"*30)
        
        # Step 2: Alignment
        if not self.load_and_align(target_size):
            return False
        
        # Step 3: Subtraction
        if not self.compute_difference():
            return False
        
        # Step 4: Thresholding
        if not self.threshold_defects():
            return False
        
        # Step 5: Noise Removal
        if not self.remove_noise():
            return False
        
        # Summary
        print("\n" + "="*60)
        print("✅ MILESTONE 1 COMPLETE!")
        print("="*60)
        print(f"All output images saved in: {os.path.abspath(self.output_dir)}")
        print("\nGenerated files:")
        print("  1. 01_template_aligned.png - Aligned template image")
        print("  2. 01_test_aligned.png - Aligned test image")
        print("  3. 02_difference_raw.png - Raw difference map")
        print("  4. 03_threshold_otsu.png - Binary threshold mask")
        print("  5. 04_noise_removed_final.png - Final clean defect mask")
        print("\n📊 These are the DELIVERABLES for Milestone 1!")
        print("="*60 + "\n")
        
        return True
    
    def visualize_results(self):
        """
        Create a visual comparison of all steps
        """
        if any(img is None for img in [self.template_gray, self.test_gray, 
                                       self.diff, self.thresh, self.clean]):
            print("❌ Pipeline not complete. Run run_pipeline() first!")
            return
        
        # Create a comparison image
        fig, axes = None, None
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            axes[0, 0].imshow(self.template_gray, cmap='gray')
            axes[0, 0].set_title("Step 1: Template (Good)")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(self.test_gray, cmap='gray')
            axes[0, 1].set_title("Step 2: Test (May have defect)")
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(self.diff, cmap='gray')
            axes[0, 2].set_title("Step 3: Difference")
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(self.thresh, cmap='gray')
            axes[1, 0].set_title("Step 4: Threshold (Otsu)")
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(self.clean, cmap='gray')
            axes[1, 1].set_title("Step 5: Final Mask (Clean)")
            axes[1, 1].axis('off')
            
            # Hide the 6th subplot
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "00_pipeline_visualization.png"), 
                       dpi=100, bbox_inches='tight')
            print("✔️ Pipeline visualization saved!")
            
        except ImportError:
            print("⚠️ matplotlib not installed. Skipping visualization.")


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    """
    Example: How to use the PCBDefectDetector
    
    Instructions:
    1. Place a template image (perfect PCB) in: dataset/template/
    2. Place a test image (PCB with defect) in: dataset/test/
    3. Run this script
    """
    
    # Paths to images
    template_image = "dataset/template/template.png"  # Change to your image
    test_image = "dataset/test/test.png"              # Change to your image
    
    # Create detector
    detector = PCBDefectDetector(
        template_path=template_image,
        test_path=test_image,
        output_dir="output"
    )
    
    # Run the pipeline
    success = detector.run_pipeline(target_size=(640, 480))
    
    if success:
        # Optional: Create visualization
        try:
            detector.visualize_results()
        except:
            pass
    else:
        print("\n❌ Pipeline failed!")
        print("Make sure you have:")
        print(f"  - Template image at: {os.path.abspath(template_image)}")
        print(f"  - Test image at: {os.path.abspath(test_image)}")
