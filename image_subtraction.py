import cv2
import numpy as np
import os


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    return img


def align_images(template, test):
    # Assume images are already aligned for simplicity
    # In a real scenario, use feature matching or homography for alignment
    return test


def subtract_and_threshold(template, test):
    # Convert to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(template_gray, test_gray)

    # Apply Gaussian blur to reduce noise
    diff_blurred = cv2.GaussianBlur(diff, (5, 5), 0)

    # Threshold using Otsu's method
    _, thresh = cv2.threshold(diff_blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    return diff, thresh, cleaned


def process_pair(template_path, test_path, output_dir):
    try:
        template = load_image(template_path)
        test = load_image(test_path)

        # Align test image to template (placeholder)
        aligned_test = align_images(template, test)

        # Subtract and threshold
        diff, thresh, cleaned = subtract_and_threshold(template, aligned_test)

        # Save results
        base_name = os.path.splitext(os.path.basename(test_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_diff.jpg"), diff)
        cv2.imwrite(os.path.join(
            output_dir, f"{base_name}_thresh.jpg"), thresh)
        cv2.imwrite(os.path.join(
            output_dir, f"{base_name}_cleaned.jpg"), cleaned)

        print(f"Processed {base_name}")

    except Exception as e:
        print(f"Error processing {test_path}: {e}")


def main():
    base_dir = "PCB_DATASET"
    template_dir = os.path.join(base_dir, "PCB_USED")
    output_dir = os.path.join(base_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Defect types
    defect_types = ["Missing_hole", "Mouse_bite",
                    "Open_circuit", "Short", "Spur", "Spurious_copper"]

    for defect_type in defect_types:
        test_dir = os.path.join(base_dir, "images", defect_type)
        if not os.path.exists(test_dir):
            print(f"Test directory not found: {test_dir}")
            continue

        for test_file in os.listdir(test_dir):
            if test_file.endswith('.jpg'):
                # Extract PCB ID (e.g., '01' from '01_missing_hole_01.jpg')
                pcb_id = test_file.split('_')[0]
                template_file = f"{pcb_id}.JPG"
                template_path = os.path.join(template_dir, template_file)
                test_path = os.path.join(test_dir, test_file)

                if os.path.exists(template_path):
                    process_pair(template_path, test_path, output_dir)
                else:
                    print(f"Template not found for {test_file}")


if __name__ == "__main__":
    main()
