"""
TEST DATA GENERATOR
Creates simple synthetic PCB images for testing Milestone 1
"""

import cv2
import numpy as np
import os

def create_test_images(output_dir="dataset"):
    """
    Create synthetic PCB images for testing
    - Template: perfect checkerboard pattern
    - Test: same pattern but with a defect (dark spot)
    """
    
    os.makedirs(os.path.join(output_dir, "template"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # Image size
    height, width = 480, 640
    
    # Create a synthetic PCB pattern (checkerboard with varying intensity)
    template = np.ones((height, width), dtype=np.uint8) * 100
    
    # Add checkerboard pattern (simulates PCB traces)
    for i in range(0, height, 40):
        for j in range(0, width, 40):
            if (i // 40 + j // 40) % 2 == 0:
                template[i:i+40, j:j+40] = 150
    
    # Add some noise for realism
    noise = np.random.normal(0, 5, (height, width)).astype(np.int16)
    template = np.clip(template.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Create test image (same as template)
    test = template.copy()
    
    # Add a defect (dark spot in the middle)
    cv2.circle(test, (320, 240), 30, 50, -1)  # Dark spot = defect
    
    # Add some lines (defect traces)
    cv2.line(test, (300, 200), (350, 280), 40, 5)
    
    # Save images
    template_path = os.path.join(output_dir, "template", "template.png")
    test_path = os.path.join(output_dir, "test", "test.png")
    
    cv2.imwrite(template_path, template)
    cv2.imwrite(test_path, test)
    
    print("✅ Test images created!")
    print(f"   Template: {template_path}")
    print(f"   Test: {test_path}")
    print("\nYou can now run the main script:")
    print("   python milestone1_pcb_defect_detection.py")


if __name__ == "__main__":
    create_test_images()
