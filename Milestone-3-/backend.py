import cv2
import numpy as np
from PIL import Image

def detect_defect(image):
    """
    PROFESSIONAL PCB DEFECT DETECTION
    Uses image processing techniques from Milestone 1 & 2
    
    Returns:
        - result_img: Image with defect boxes drawn
        - defect_info: Dictionary with detection results
    """
    
    # ===== STEP 1: Prepare Image =====
    # Convert PIL to OpenCV format
    img = np.array(image)
    if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    original_img = img.copy()
    height, width = img.shape[:2]
    
    # Resize for processing (standard size)
    img_resized = cv2.resize(img, (640, 480))
    
    # ===== STEP 2: Convert to Grayscale =====
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ===== STEP 3: Apply Thresholding (Binary Image) =====
    # This separates defects from background
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # ===== STEP 4: Morphological Operations (Clean up image) =====
    # Remove noise and fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    # ===== STEP 5: Find Contours (Defect Edges) =====
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ===== STEP 6: Filter and Draw Valid Defects =====
    img_with_boxes = img_resized.copy()
    defects_found = []
    min_area = 50  # Minimum defect size (pixels²)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Only consider significant defects
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Store defect info
            defects_found.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area
            })
            
            # Draw GREEN rectangle for detected defect
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw contour
            cv2.drawContours(img_with_boxes, [contour], 0, (0, 165, 255), 2)
            
            # Add label
            label = f"Area:{int(area)}px"
            cv2.putText(img_with_boxes, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # ===== STEP 7: Prepare Results =====
    # Resize result back to original size for display
    result_img = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    
    # Create detailed report
    if len(defects_found) > 0:
        defect_info = {
            'status': 'DEFECT DETECTED',
            'count': len(defects_found),
            'defects': defects_found,
            'confidence': 'HIGH' if len(defects_found) > 0 else 'NONE'
        }
    else:
        defect_info = {
            'status': 'NO DEFECT',
            'count': 0,
            'defects': [],
            'confidence': 'NONE'
        }
    
    return result_img, defect_info
