import cv2
import numpy as np
import os


def load_image(path):
    img = cv2.imread(path, 0)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    return img


def align_images(template, test):
    """
    Align test image to template using ORB + Homography.
    """
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(test, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        print("Not enough matches — skipping alignment.")
        return test

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
    aligned = cv2.warpPerspective(
        test, M, (template.shape[1], template.shape[0]))

    return aligned


def subtract_and_threshold(template, test):
    # Compute absolute difference
    diff = cv2.absdiff(template, test)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(diff, (5, 5), 0)

    # Threshold using Otsu's method
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return diff, thresh


def highlight_defects(test_img, mask):
    """
    Highlights defect areas in red on the test image.
    """
    test_color = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    test_color[mask == 255] = [0, 0, 255]
    return test_color


def process_pair(template_path, test_path, output_dir):
    try:
        template = load_image(template_path)
        test = load_image(test_path)

        # Align test image to template (placeholder)
        aligned_test = align_images(template, test)

        # Subtract and threshold
        diff, thresh = subtract_and_threshold(template, aligned_test)

        # Highlight defects
        highlighted = highlight_defects(aligned_test, thresh)

        # Save results
        base_name = os.path.splitext(os.path.basename(test_path))[0]
        cv2.imwrite(os.path.join(
            output_dir, f"{base_name}_aligned.jpg"), aligned_test)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_diff.jpg"), diff)
        cv2.imwrite(os.path.join(
            output_dir, f"{base_name}_thresh.jpg"), thresh)
        cv2.imwrite(os.path.join(
            output_dir, f"{base_name}_highlighted.jpg"), highlighted)

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
