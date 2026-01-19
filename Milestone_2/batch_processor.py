"""
MILESTONE 2: Batch Process Multiple Images
Process multiple defect images with their annotations
"""

import os
import sys
from pathlib import Path
from milestone2_defect_localization import DefectLocalizer


class BatchDefectProcessor:
    """Process multiple images with annotations"""
    
    def __init__(self, dataset_dir, num_images=5):
        self.dataset_dir = dataset_dir
        self.num_images = num_images
        self.results = []
    
    def process_dataset(self):
        """
        Process multiple image pairs from DeepPCB dataset
        """
        print("\n" + "="*60)
        print("MILESTONE 2: BATCH PROCESSING")
        print("="*60)
        
        images_dir = os.path.join(self.dataset_dir, "images")
        annotations_dir = os.path.join(self.dataset_dir, "Annotations")
        
        # Get all defect types
        defect_types = sorted([d for d in os.listdir(images_dir) 
                              if os.path.isdir(os.path.join(images_dir, d))])
        
        print(f"[OK] Found {len(defect_types)} defect types")
        
        image_count = 0
        
        # Process each defect type
        for defect_type in defect_types:
            defect_img_dir = os.path.join(images_dir, defect_type)
            defect_ann_dir = os.path.join(annotations_dir, defect_type)
            
            # Get all images in this defect type
            images = sorted([f for f in os.listdir(defect_img_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # Process up to num_images total
            for img_file in images:
                if image_count >= self.num_images:
                    break
                
                # Find corresponding XML
                xml_file = Path(img_file).stem + ".xml"
                
                img_path = os.path.join(defect_img_dir, img_file)
                xml_path = os.path.join(defect_ann_dir, xml_file)
                
                # Check if both exist
                if not os.path.exists(xml_path):
                    continue
                
                print(f"\n{'='*60}")
                print(f"PROCESSING IMAGE {image_count + 1}/{self.num_images}")
                print(f"{'='*60}")
                print(f"Type: {defect_type}")
                print(f"Image: {img_file}")
                
                # Create output directory
                output_dir = f"output/image_{image_count:02d}_{defect_type}"
                
                # Process
                localizer = DefectLocalizer(
                    image_path=img_path,
                    xml_path=xml_path,
                    output_dir=output_dir
                )
                
                if localizer.run_pipeline():
                    image_count += 1
                    self.results.append({
                        'index': image_count,
                        'type': defect_type,
                        'image': img_file,
                        'status': 'SUCCESS',
                        'output_dir': output_dir,
                        'roi_count': len(localizer.roi_list)
                    })
                else:
                    self.results.append({
                        'index': image_count,
                        'type': defect_type,
                        'image': img_file,
                        'status': 'FAILED',
                        'output_dir': output_dir,
                        'roi_count': 0
                    })
            
            if image_count >= self.num_images:
                break
        
        self._print_summary()
    
    def _print_summary(self):
        """Print processing summary"""
        print("\n" + "="*60)
        print("[COMPLETE] BATCH PROCESSING COMPLETE!")
        print("="*60)
        print(f"Total images processed: {len(self.results)}")
        
        successful = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        failed = sum(1 for r in self.results if r['status'] == 'FAILED')
        total_rois = sum(r['roi_count'] for r in self.results)
        
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total ROIs extracted: {total_rois}")
        
        print("\nDetailed Results:")
        for r in self.results:
            status_icon = "[OK]" if r['status'] == 'SUCCESS' else "[FAIL]"
            print(f"{status_icon} Image {r['index']}: {r['type']} - {r['image']}")
            if r['status'] == 'SUCCESS':
                print(f"   -> {r['roi_count']} defect(s) found")
                print(f"   -> Output: {r['output_dir']}")
        
        print("="*60 + "\n")


# ============ USAGE ============
if __name__ == "__main__":
    
    dataset_dir = r"C:\Users\Vishwa Adhesh\Downloads\PCB_DATASET"
    
    processor = BatchDefectProcessor(
        dataset_dir=dataset_dir,
        num_images=10  # Process 10 images
    )
    
    processor.process_dataset()
    
    print("[OK] All images processed!")
    print("[OK] Check 'output/' folder for results")
