"""
MILESTONE 1: PCB Defect Detection - DEEPPCB Dataset Handler
Automatic processing of DeepPCB dataset structure
"""

import cv2
import os
import numpy as np
from pathlib import Path

class PCBDatasetHandler:
    """
    Handler for DeepPCB dataset structure
    Automatically finds and pairs template/test images
    """
    
    @staticmethod
    def find_dataset_pairs(dataset_dir="C:\\Users\\Vishwa Adhesh\\Downloads\\PCB_DATASET"):
        """
        Scan the DeepPCB dataset and find template/test pairs
        
        DeepPCB structure:
        - PCB_USED/ contains template PCBs (perfect)
        - images/ contains defect types with test images
        """
        
        print("\n" + "="*60)
        print("SCANNING DEEPPCB DATASET")
        print("="*60)
        
        pairs = []
        
        # Path to templates
        pcb_used_dir = os.path.join(dataset_dir, "PCB_USED")
        images_dir = os.path.join(dataset_dir, "images")
        
        if not os.path.exists(images_dir):
            print(f"❌ Dataset not found at {dataset_dir}")
            return []
        
        # Get list of templates
        templates = sorted([f for f in os.listdir(pcb_used_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"✔️ Found {len(templates)} template PCBs")
        
        # Scan defect types
        defect_types = [d for d in os.listdir(images_dir) 
                       if os.path.isdir(os.path.join(images_dir, d))]
        
        print(f"✔️ Found {len(defect_types)} defect types")
        
        # For each defect type, create pairs
        pair_count = 0
        for defect_type in defect_types:
            defect_dir = os.path.join(images_dir, defect_type)
            test_images = sorted([f for f in os.listdir(defect_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            print(f"   - {defect_type}: {len(test_images)} images")
            
            # Pair templates with test images
            for i, test_img in enumerate(test_images):
                template_idx = i % len(templates)  # Cycle through templates
                template = templates[template_idx]
                
                pair = {
                    'template': os.path.join(pcb_used_dir, template),
                    'test': os.path.join(defect_dir, test_img),
                    'defect_type': defect_type,
                    'template_name': template,
                    'test_name': test_img
                }
                pairs.append(pair)
                pair_count += 1
        
        print(f"✔️ Total pairs created: {pair_count}")
        return pairs


class PCBDefectDetector:
    """PCB Defect Detection Pipeline"""
    
    def __init__(self, template_path, test_path, output_dir="output", pair_info=None):
        self.template_path = template_path
        self.test_path = test_path
        self.output_dir = output_dir
        self.pair_info = pair_info or {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.template = None
        self.test = None
        self.template_gray = None
        self.test_gray = None
        self.diff = None
        self.thresh = None
        self.clean = None
    
    def load_and_align(self, target_size=(640, 480)):
        """STEP 2: Load and align images"""
        try:
            self.template = cv2.imread(self.template_path)
            self.test = cv2.imread(self.test_path)
            
            if self.template is None or self.test is None:
                return False
            
            self.template = cv2.resize(self.template, target_size)
            self.test = cv2.resize(self.test, target_size)
            
            self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            self.test_gray = cv2.cvtColor(self.test, cv2.COLOR_BGR2GRAY)
            
            return True
        except Exception as e:
            print(f"❌ Error loading images: {str(e)}")
            return False
    
    def compute_difference(self):
        """STEP 3: Compute difference"""
        try:
            self.diff = cv2.absdiff(self.template_gray, self.test_gray)
            return True
        except Exception as e:
            print(f"❌ Error computing difference: {str(e)}")
            return False
    
    def threshold_defects(self):
        """STEP 4: Threshold"""
        try:
            otsu_threshold, self.thresh = cv2.threshold(
                self.diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return True
        except Exception as e:
            print(f"❌ Error thresholding: {str(e)}")
            return False
    
    def remove_noise(self, kernel_size=(3, 3)):
        """STEP 5: Remove noise"""
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            self.clean = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel)
            return True
        except Exception as e:
            print(f"❌ Error removing noise: {str(e)}")
            return False
    
    def save_results(self, pair_index=0):
        """Save all processing results"""
        try:
            prefix = f"pair_{pair_index:03d}"
            
            cv2.imwrite(os.path.join(self.output_dir, f"{prefix}_01_template.png"), 
                       self.template_gray)
            cv2.imwrite(os.path.join(self.output_dir, f"{prefix}_02_test.png"), 
                       self.test_gray)
            cv2.imwrite(os.path.join(self.output_dir, f"{prefix}_03_difference.png"), 
                       self.diff)
            cv2.imwrite(os.path.join(self.output_dir, f"{prefix}_04_threshold.png"), 
                       self.thresh)
            cv2.imwrite(os.path.join(self.output_dir, f"{prefix}_05_final_mask.png"), 
                       self.clean)
            
            return True
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
            return False
    
    def run_pipeline(self, target_size=(640, 480)):
        """Run complete pipeline"""
        if not self.load_and_align(target_size):
            return False
        if not self.compute_difference():
            return False
        if not self.threshold_defects():
            return False
        if not self.remove_noise():
            return False
        return True


def process_deeppcb_dataset(dataset_dir="C:\\Users\\Vishwa Adhesh\\Downloads\\PCB_DATASET", 
                           num_pairs=5, output_base_dir="output"):
    """
    Process DeepPCB dataset - analyze multiple image pairs
    """
    
    print("\n" + "🔵"*30)
    print("MILESTONE 1: DEEPPCB DATASET PROCESSING")
    print("🔵"*30)
    
    # Find all pairs
    pairs = PCBDatasetHandler.find_dataset_pairs(dataset_dir)
    
    if not pairs:
        print("❌ No dataset pairs found!")
        return False
    
    # Process first N pairs
    num_to_process = min(num_pairs, len(pairs))
    
    for i in range(num_to_process):
        pair = pairs[i]
        
        print(f"\n{'='*60}")
        print(f"PROCESSING PAIR {i+1}/{num_to_process}")
        print(f"{'='*60}")
        print(f"Template: {pair['template_name']}")
        print(f"Test: {pair['test_name']}")
        print(f"Defect Type: {pair['defect_type']}")
        
        # Create output dir for this pair
        pair_output_dir = os.path.join(output_base_dir, f"pair_{i:03d}_{pair['defect_type']}")
        
        # Create detector and run pipeline
        detector = PCBDefectDetector(
            template_path=pair['template'],
            test_path=pair['test'],
            output_dir=pair_output_dir,
            pair_info=pair
        )
        
        if detector.run_pipeline():
            detector.save_results(i)
            print(f"✅ Pair {i+1} processed successfully")
            print(f"   Results saved to: {pair_output_dir}")
            
            # Print statistics
            defect_pixels = np.count_nonzero(detector.clean)
            total_pixels = detector.clean.shape[0] * detector.clean.shape[1]
            defect_percentage = (defect_pixels / total_pixels) * 100
            print(f"   Defect area: {defect_pixels} pixels ({defect_percentage:.2f}%)")
        else:
            print(f"❌ Failed to process pair {i+1}")
    
    print("\n" + "="*60)
    print("✅ MILESTONE 1 COMPLETE!")
    print("="*60)
    print(f"Processed {num_to_process} image pairs")
    print(f"Results saved in: {os.path.abspath(output_base_dir)}")
    print("="*60 + "\n")
    
    return True


# ============ USAGE ============
if __name__ == "__main__":
    
    # Process 5 pairs from the dataset
    success = process_deeppcb_dataset(
        dataset_dir=r"C:\Users\Vishwa Adhesh\Downloads\PCB_DATASET",
        num_pairs=5,
        output_base_dir="output"
    )
    
    if success:
        print("✅ All pipelines completed!")
    else:
        print("❌ Processing failed!")
