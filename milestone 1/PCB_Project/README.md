# 🌱 MILESTONE 1: PCB Defect Detection

## What's Inside?

This folder contains the complete implementation for **Milestone 1: Dataset Preparation & Image Processing**.

### Folder Structure
```
PCB_Project/
├── dataset/
│   ├── template/          ← Put perfect PCB images here
│   └── test/              ← Put PCB images with defects here
├── output/                ← Results will be saved here
└── milestone1_pcb_defect_detection.py  ← Main script (DO NOT EDIT)
```

## 🚀 How to Use

### Step 1: Get Test Images

1. Download the **DeepPCB dataset** or use your own PCB images
2. Place:
   - **Perfect PCB image** → `dataset/template/template.png`
   - **PCB with defect** → `dataset/test/test.png`

### Step 2: Install Required Libraries

```bash
pip install opencv-python numpy matplotlib
```

### Step 3: Run the Script

```bash
python milestone1_pcb_defect_detection.py
```

### Step 4: Check Results

All output images will be saved in the `output/` folder:

```
output/
├── 00_pipeline_visualization.png    ← See all steps at once!
├── 01_template_aligned.png          ← Step 2 output
├── 01_test_aligned.png              ← Step 2 output
├── 02_difference_raw.png            ← Step 3 output
├── 03_threshold_otsu.png            ← Step 4 output
└── 04_noise_removed_final.png       ← Step 5 output (FINAL MASK!)
```

## 📋 What Each Step Does

| Step | Name | Input | Output | Purpose |
|------|------|-------|--------|---------|
| 2 | Alignment | Color images | Grayscale, same size | Make images comparable |
| 3 | Subtraction | 2 grayscale images | Difference map | Find what's different |
| 4 | Threshold | Difference map | Black & white | Make defects clear |
| 5 | Noise Removal | Binary image | Clean mask | Remove random noise |

## ✅ Milestones Deliverables

When Milestone 1 is DONE, you have:

✔️ **Aligned dataset** → `01_template_aligned.png` + `01_test_aligned.png`
✔️ **Subtraction image** → `02_difference_raw.png`
✔️ **Thresholded defect image** → `03_threshold_otsu.png`
✔️ **Noise-free defect mask** → `04_noise_removed_final.png`

## 🧪 Testing

The script will:
1. Load both images ✅
2. Resize and convert to grayscale ✅
3. Compute pixel-by-pixel difference ✅
4. Apply Otsu's automatic thresholding ✅
5. Remove noise using morphological operations ✅
6. Save all 5 output images ✅

All steps are **fully explained in the code comments** for learning!

## ❓ Need Help?

Check the output images:
- If **template and test** are similar in size → Step 2 ✅
- If **difference image** shows bright spots where defects are → Step 3 ✅
- If **threshold image** is pure black & white → Step 4 ✅
- If **final mask** is clean without noise → Step 5 ✅

---

**Next Step:** After confirming all outputs look correct, I'll give you **Milestone 2: Contour Detection** 📦
