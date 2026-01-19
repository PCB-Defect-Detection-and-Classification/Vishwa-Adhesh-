"""
MILESTONE 2: Defect Localization using Contours
Step-by-step: Draw boxes around defects and crop them

Goal: "Draw a box around the defect so the computer knows where the problem is"
"""

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from pathlib import Path


class XMLAnnotationParser:
    """
    Parse XML annotations from DeepPCB dataset
    Extract bounding boxes for each defect
    """
    
    @staticmethod
    def parse_xml(xml_path):
        """
        STEP 1: Load annotation (bounding boxes)
        
        What is annotation?
        -> A file that says "defect is in this rectangle"
        -> Rectangle has: x_min, y_min, x_max, y_max
        
        Returns:
            List of bounding boxes: [(x_min, y_min, x_max, y_max), ...]
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            bboxes = []
            
            # Find all object elements (each defect)
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                
                x_min = int(bndbox.find('xmin').text)
                y_min = int(bndbox.find('ymin').text)
                x_max = int(bndbox.find('xmax').text)
                y_max = int(bndbox.find('ymax').text)
                
                bboxes.append((x_min, y_min, x_max, y_max))
            
            return bboxes
        
        except Exception as e:
            print(f"[FAIL] Error parsing XML: {str(e)}")
            return []


class DefectLocalizer:
    """
    Complete Milestone 2 pipeline:
    1. Load image + annotations
    2. Create defect mask
    3. Find contours
    4. Draw bounding boxes
    5. Crop defect regions (ROI)
    """
    
    def __init__(self, image_path, xml_path, output_dir="output"):
        """
        Initialize with image and annotation paths
        
        Args:
            image_path: Path to PCB image
            xml_path: Path to XML annotation
            output_dir: Where to save results
        """
        self.image_path = image_path
        self.xml_path = xml_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Will store images at each step
        self.original_img = None
        self.img_resized = None
        self.mask = None
        self.contours = None
        self.img_with_boxes = None
        self.roi_list = []
        self.bboxes = []
    
    # ============ STEP 1: LOAD IMAGE & ANNOTATION ============
    def load_image_and_annotation(self, target_size=(640, 480)):
        """
        STEP 1: Load the image and parse annotation
        
        Baby explanation:
        "Read the picture and its label (where the mistake is)"
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 1: LOAD IMAGE & ANNOTATION")
        print("="*60)
        
        try:
            # Load image
            self.original_img = cv2.imread(self.image_path)
            
            if self.original_img is None:
                print(f"[FAIL] Cannot load image from {self.image_path}")
                return False
            
            print(f"[OK] Image loaded: {self.original_img.shape}")
            
            # Resize image
            self.img_resized = cv2.resize(self.original_img, target_size)
            print(f"[OK] Image resized to: {target_size}")
            
            # Parse annotation
            self.bboxes = XMLAnnotationParser.parse_xml(self.xml_path)
            
            if not self.bboxes:
                print(f"[WARN] No defects found in annotation")
                return False
            
            print(f"[OK] Found {len(self.bboxes)} defect(s) in annotation")
            for i, (x1, y1, x2, y2) in enumerate(self.bboxes):
                print(f"   Defect {i+1}: ({x1}, {y1}) to ({x2}, {y2})")
            
            print("[DONE] STEP 1 COMPLETE: Image & annotation loaded\n")
            return True
            
        except Exception as e:
            print(f"[FAIL] Error in Step 1: {str(e)}")
            return False
    
    # ============ STEP 2: CREATE DEFECT MASK ============
    def create_defect_mask(self):
        """
        STEP 2: Create a mask showing defect locations
        
        Baby explanation:
        "Paint defect area WHITE, rest BLACK"
        
        What is a mask?
        -> White = defect
        -> Black = no defect
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 2: CREATE DEFECT MASK")
        print("="*60)
        
        if self.img_resized is None:
            print("[FAIL] Error: Image not loaded yet!")
            return False
        
        try:
            # Get image dimensions
            height, width = self.img_resized.shape[:2]
            
            # Create empty black mask
            self.mask = np.zeros((height, width), dtype="uint8")
            print(f"[OK] Created empty mask: {self.mask.shape}")
            
            # Scale bounding boxes to resized image
            original_h, original_w = self.original_img.shape[:2]
            scale_x = width / original_w
            scale_y = height / original_h
            
            # Draw defect rectangles as WHITE on the mask
            scaled_bboxes = []
            for x1, y1, x2, y2 in self.bboxes:
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                # Draw white rectangle
                cv2.rectangle(self.mask, (x1_scaled, y1_scaled), 
                             (x2_scaled, y2_scaled), 255, -1)
                
                scaled_bboxes.append((x1_scaled, y1_scaled, x2_scaled, y2_scaled))
            
            self.bboxes = scaled_bboxes
            
            # Count white pixels
            white_pixels = np.count_nonzero(self.mask)
            total_pixels = self.mask.shape[0] * self.mask.shape[1]
            defect_percentage = (white_pixels / total_pixels) * 100
            
            print(f"[OK] Defect area marked: {white_pixels} pixels ({defect_percentage:.2f}%)")
            
            # Save mask
            cv2.imwrite(os.path.join(self.output_dir, "02_defect_mask.png"), 
                       self.mask)
            
            print(f"[OK] Mask saved to {self.output_dir}")
            print("[DONE] STEP 2 COMPLETE: Defect mask created\n")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Error in Step 2: {str(e)}")
            return False
    
    # ============ STEP 3: FIND CONTOURS ============
    def find_contours(self):
        """
        STEP 3: Find defect shapes using contours
        
        What is a contour?
        -> "The border/outline of the defect"
        
        Why?
        -> Computer can now understand defect shape
        
        Baby explanation:
        "Draw the border of each defect"
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 3: FIND CONTOURS (DEFECT BORDERS)")
        print("="*60)
        
        if self.mask is None:
            print("[FAIL] Error: Mask not created yet!")
            return False
        
        try:
            # Find contours in the mask
            self.contours, _ = cv2.findContours(
                self.mask, 
                cv2.RETR_EXTERNAL,  # Get only outer contours
                cv2.CHAIN_APPROX_SIMPLE  # Simplify contour paths
            )
            
            print(f"[OK] Found {len(self.contours)} contour(s)")
            
            # Analyze each contour
            for i, cnt in enumerate(self.contours):
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                print(f"   Contour {i+1}: Area={area:.0f}px, Perimeter={perimeter:.0f}px")
            
            print("[DONE] STEP 3 COMPLETE: Contours detected\n")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Error in Step 3: {str(e)}")
            return False
    
    # ============ STEP 4: DRAW BOUNDING BOXES ============
    def draw_bounding_boxes(self):
        """
        STEP 4: Draw boxes around defects
        
        Why?
        -> Boxes are easy for ML models to understand
        -> We can quickly see where defects are
        
        Baby explanation:
        "Draw a GREEN box around each defect"
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 4: DRAW BOUNDING BOXES")
        print("="*60)
        
        if self.img_resized is None:
            print("[FAIL] Error: Image not loaded yet!")
            return False
        
        try:
            # Copy image for drawing
            self.img_with_boxes = self.img_resized.copy()
            
            # Draw green boxes around defects
            for i, (x1, y1, x2, y2) in enumerate(self.bboxes):
                # Draw rectangle with GREEN color (BGR: 0, 255, 0)
                cv2.rectangle(self.img_with_boxes, (x1, y1), (x2, y2), 
                             (0, 255, 0), 2)
                
                # Add text label
                cv2.putText(self.img_with_boxes, f"Defect {i+1}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
            
            print(f"[OK] Drew {len(self.bboxes)} bounding box(es)")
            
            # Save image with boxes
            cv2.imwrite(os.path.join(self.output_dir, "03_bounding_boxes.png"), 
                       self.img_with_boxes)
            
            print(f"[OK] Image with boxes saved to {self.output_dir}")
            print("[DONE] STEP 4 COMPLETE: Bounding boxes drawn\n")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Error in Step 4: {str(e)}")
            return False
    
    # ============ STEP 5: CROP DEFECT REGIONS (ROI) ============
    def crop_roi(self):
        """
        STEP 5: Crop and extract defect regions
        
        What is ROI?
        -> ROI = Region Of Interest
        -> Just the defect area, nothing else
        
        Why?
        -> ML models learn better from close-up defect images
        -> Faster and more efficient
        
        Baby explanation:
        "Cut out just the mistake from the picture"
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("STEP 5: CROP DEFECT REGIONS (ROI)")
        print("="*60)
        
        if self.img_resized is None:
            print("[FAIL] Error: Image not loaded yet!")
            return False
        
        try:
            self.roi_list = []
            
            for i, (x1, y1, x2, y2) in enumerate(self.bboxes):
                # Extract region of interest
                roi = self.img_resized[y1:y2, x1:x2]
                
                # Check if ROI is valid
                if roi.size == 0:
                    print(f"[WARN] ROI {i+1} is empty, skipping...")
                    continue
                
                self.roi_list.append(roi)
                
                # Save ROI
                roi_filename = f"05_roi_{i+1:02d}.png"
                cv2.imwrite(os.path.join(self.output_dir, roi_filename), roi)
                
                print(f"[OK] ROI {i+1} cropped and saved: {roi.shape}")
            
            print(f"[OK] Total {len(self.roi_list)} ROI(s) extracted")
            print(f"[OK] ROI images saved to {self.output_dir}")
            print("[DONE] STEP 5 COMPLETE: Defect regions cropped\n")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Error in Step 5: {str(e)}")
            return False
    
    # ============ SAVE SUMMARY ============
    def save_original_image(self):
        """Save resized original image for reference"""
        try:
            cv2.imwrite(os.path.join(self.output_dir, "01_original_resized.png"), 
                       self.img_resized)
        except:
            pass
    
    # ============ MAIN PIPELINE ============
    def run_pipeline(self, target_size=(640, 480)):
        """
        Run complete Milestone 2 pipeline
        """
        print("\n" + "="*60)
        print("MILESTONE 2: DEFECT LOCALIZATION")
        print("="*60)
        
        # Step 1: Load
        if not self.load_image_and_annotation(target_size):
            return False
        
        # Save original
        self.save_original_image()
        
        # Step 2: Create mask
        if not self.create_defect_mask():
            return False
        
        # Step 3: Find contours
        if not self.find_contours():
            return False
        
        # Step 4: Draw boxes
        if not self.draw_bounding_boxes():
            return False
        
        # Step 5: Crop ROI
        if not self.crop_roi():
            return False
        
        # Summary
        print("\n" + "="*60)
        print("[DONE] MILESTONE 2 COMPLETE!")
        print("="*60)
        print(f"Output saved in: {os.path.abspath(self.output_dir)}")
        print("\nGenerated files:")
        print("  1. 01_original_resized.png - Resized original image")
        print("  2. 02_defect_mask.png - Defect mask (white=defect)")
        print("  3. 03_bounding_boxes.png - Image with green boxes")
        print("  4. 05_roi_01.png, 05_roi_02.png, ... - Cropped defect regions")
        print("\nThese are the DELIVERABLES for Milestone 2!")
        print("="*60 + "\n")
        
        return True


# ============ USAGE ============
if __name__ == "__main__":
    """
    Example: Process a single image with annotations
    """
    
    # Example 1: Single image processing
    image_path = r"C:\Users\Vishwa Adhesh\Downloads\PCB_DATASET\images\Missing_hole\01_missing_hole_01.jpg"
    xml_path = r"C:\Users\Vishwa Adhesh\Downloads\PCB_DATASET\Annotations\Missing_hole\01_missing_hole_01.xml"
    
    # Create localizer
    localizer = DefectLocalizer(
        image_path=image_path,
        xml_path=xml_path,
        output_dir="output/sample_defect"
    )
    
    # Run pipeline
    success = localizer.run_pipeline(target_size=(640, 480))
    
    if success:
        print("[OK] Pipeline completed successfully!")
    else:
        print("[FAIL] Pipeline failed!")
