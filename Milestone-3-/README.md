# 🔬 Milestone 3: PCB Defect Detection System
## Frontend + Backend Integration

### 📋 Overview
Professional PCB (Printed Circuit Board) defect detection system using:
- **Frontend**: Streamlit (Web UI)
- **Backend**: OpenCV + Image Processing
- **Deployment**: Python 3.8+

### 🚀 Quick Start

#### 1. **Clone & Setup**
```bash
git clone <repo-url>
cd Milestone_3
python -m venv venv
```

#### 2. **Activate Virtual Environment**
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

#### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

#### 4. **Run Application**
```bash
streamlit run app.py
```

🔗 Open: `http://localhost:8501`

---

### 📁 File Structure
```
Milestone_3/
├── app.py              # Frontend UI (Streamlit)
├── backend.py          # Detection Logic (OpenCV)
├── requirements.txt    # Dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

---

### ✨ Features

✅ **Upload PCB Images** (JPG, PNG, JPEG)
✅ **Real-time Detection** using thresholding & contours
✅ **Defect Analysis**:
   - Defect count
   - Position coordinates
   - Size (width × height)
   - Area in pixels²
   
✅ **Visual Results**:
   - Green bounding boxes around defects
   - Contour highlighting
   - Professional UI with sidebar

---

### 🔧 Technical Details

**Detection Method:**
1. Convert to grayscale
2. Binary thresholding
3. Morphological operations (closing/opening)
4. Contour detection
5. Filter by minimum area
6. Draw bounding boxes

**Requirements:**
- `streamlit` - Web framework
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `pillow` - Image handling

---

### 📊 Sample Results
- **Input:** PCB image with defects
- **Output:** 
  - Marked image with green boxes
  - Defect count & details
  - Confidence level

---

### 👨‍💼 Project Info
- **Infosys Springboard Project**
- **Milestone 3**: Frontend + Backend Integration
- **Status**: ✅ Complete

---

### 📝 Notes
- Minimum defect area: 50 pixels²
- Processing resolution: 640×480
- Real-time analysis
- No GPU required

---

### 🎓 Learning Outcomes
By completing this milestone, you will learn:
- ✅ Building web interfaces with Streamlit
- ✅ Image processing with OpenCV
- ✅ Frontend-Backend integration
- ✅ Deploying Python applications

---

### 📧 Support
For issues or questions, contact the project team.

**Last Updated:** January 2026
