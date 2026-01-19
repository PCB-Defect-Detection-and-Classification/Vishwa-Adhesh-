# 🌿 MILESTONE 2: Defect Localization using Contours

## 🎯 Goal (Very Simple Words)

> "Draw a box around the defect so the computer knows where the problem is."

Think like: 🖍️ Teacher circling the mistake in a notebook.

---

## 📚 What You'll Learn

This milestone shows how to:
1. **Load image + annotation together** - Read the picture and its label
2. **Create defect mask** - Mark defect areas in white
3. **Find contours** - Detect the border of defects
4. **Draw bounding boxes** - Put green boxes around defects
5. **Crop ROI regions** - Cut out just the defect part

---

## 🧩 The 5 Steps (Very Simple)

### STEP 1: Load Image + Annotation
```
Input:  Image file + XML annotation
Output: Image loaded, defect locations found
```
✔️ Read the picture  
✔️ Read where the mistake is  

### STEP 2: Create Defect Mask
```
Input:  Image + Bounding boxes from annotation
Output: Black & white mask (white = defect)
```
✔️ Paint defect area WHITE  
✔️ Paint background BLACK  

### STEP 3: Find Contours
```
Input:  Defect mask
Output: Contour points (defect borders)
```
✔️ Computer finds the border of each defect  

### STEP 4: Draw Bounding Boxes
```
Input:  Original image + Bounding boxes
Output: Image with green boxes drawn
```
✔️ Draw GREEN boxes around defects  
✔️ Easy for ML models to understand  

### STEP 5: Crop Defect Regions (ROI)
```
Input:  Image + Bounding boxes
Output: Cropped images of just the defects
```
✔️ Cut out defect areas  
✔️ Save as separate images  

---

## 🚀 How to Run

### Option 1: Process Single Image
```bash
python milestone2_defect_localization.py
```

### Option 2: Batch Process Multiple Images
```bash
python batch_processor.py
```

---

## 📂 Input Files

**From DeepPCB Dataset:**

```
C:\Users\Vishwa Adhesh\Downloads\PCB_DATASET\
├── images/
│   ├── Missing_hole/          ← Images with missing hole defect
│   ├── Mouse_bite/            ← Images with mouse bite defect
│   └── ...                    ← Other defect types
└── Annotations/
    ├── Missing_hole/          ← XML files with bounding boxes
    ├── Mouse_bite/
    └── ...
```

---

## 📤 Output Files

Each processed image generates:

```
output/image_XX_DEFECT_TYPE/
├── 01_original_resized.png      ← Input image (resized)
├── 02_defect_mask.png           ← White mask of defects
├── 03_bounding_boxes.png        ← Image with green boxes
├── 05_roi_01.png                ← Cropped defect region 1
├── 05_roi_02.png                ← Cropped defect region 2
└── ...                          ← More ROIs if multiple defects
```

---

## ✅ Checkpoints

**After Step 1:**
- ✔️ Image loaded?
- ✔️ Annotation parsed?
- ✔️ Defect locations found?

**After Step 2:**
- ✔️ Mask created?
- ✔️ Defect area WHITE?
- ✔️ Background BLACK?

**After Step 3:**
- ✔️ Contours detected?
- ✔️ Borders found?

**After Step 4:**
- ✔️ Green boxes drawn?
- ✔️ Boxes covering defects?

**After Step 5:**
- ✔️ ROI images created?
- ✔️ Close-up defect crops saved?

---

## 📊 Expected Output

**Input Image:**
- PCB photo with defect (maybe hard to see with naked eye)

**Output 1 - Defect Mask:**
- All black except white rectangles where defects are

**Output 2 - Bounding Boxes:**
- Original image with GREEN boxes around each defect
- Text labels "Defect 1", "Defect 2", etc.

**Output 3 - ROI Crops:**
- Small images showing only the defect areas
- Useful for training ML models

---

## 🧪 Test with Sample

The script automatically uses:
```
Image:      01_missing_hole_01.jpg
Annotation: 01_missing_hole_01.xml
Output:     output/sample_defect/
```

Just run: `python milestone2_defect_localization.py`

---

## 📋 Deliverables for Milestone 2

✅ **Defect mask** - White regions showing where defects are  
✅ **Contours detected** - Borders of defects found  
✅ **Bounding boxes drawn** - Green boxes around defects  
✅ **Cropped ROI images** - Close-up shots of defects  

---

## 🔧 Understanding the Code

### Loading Annotation (XML)
```python
bboxes = XMLAnnotationParser.parse_xml("annotation.xml")
# Returns: [(x1, y1, x2, y2), (x1, y1, x2, y2), ...]
```

### Creating Mask
```python
mask = np.zeros((height, width), dtype="uint8")
cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)  # Fill with white
```

### Finding Contours
```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### Drawing Boxes
```python
cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)  # Green box
```

### Cropping ROI
```python
roi = image[y1:y2, x1:x2]  # Extract rectangle
```

---

## ❓ Troubleshooting

**"No defects found"**
- Check if annotation file exists
- Verify XML file is in correct format

**"Empty ROI"**
- Bounding box might be too small
- Check annotation coordinates

**"Permission denied"**
- Make sure output folder is writable
- Check file paths

---

## 🎯 Next Steps

After Milestone 2:
- ✅ You can locate defects
- ✅ You can extract defect regions
- 📦 Ready for Milestone 3: Classification (if it exists!)

---

**Milestone 2: COMPLETE** ✅
