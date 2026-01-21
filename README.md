---

# 🧠 PCB Defect Detection and Classification System

## 📌 Project Overview

This project implements an **automated defect detection and classification system for Printed Circuit Boards (PCBs)** using **image processing** and **deep learning** techniques.

The system compares a **defect-free template PCB image** with a **test PCB image**, detects defects using image subtraction and contour detection, and classifies defects using a **CNN-based deep learning model**. A **web-based application** allows users to upload images and visualize results.

---

## 🎯 Objectives

* Detect and localize PCB defects automatically
* Classify defects into predefined categories
* Train and evaluate a deep learning model for high accuracy
* Build an interactive web application for image upload and results
* Export annotated images and prediction logs

---

## 🗂 Dataset

* **Dataset Name:** DeepPCB Dataset
* **Type:** Reference-based PCB defect dataset
* **Input:** Template image + Test image
* **Output:** Defect masks, bounding boxes, defect labels

---

## 🛠 Tech Stack

| Area               | Tools / Libraries                |
| ------------------ | -------------------------------- |
| Image Processing   | OpenCV, NumPy                    |
| Deep Learning      | PyTorch / TensorFlow             |
| Model Architecture | CNN / EfficientNet               |
| Frontend           | Streamlit / HTML, CSS            |
| Backend            | Python                           |
| Evaluation         | Accuracy, Loss, Confusion Matrix |
| Export             | CSV logs, Annotated images       |

---

## 📁 Project Folder Structure

```
PCB-Defect-Detection/
│
├── dataset/
│   ├── template/
│   ├── test/
│   └── labels/
│
├── preprocessing/
│   ├── subtraction.py
│   ├── thresholding.py
│   └── contour_detection.py
│
├── model/
│   ├── train.py
│   ├── evaluate.py
│   └── pcb_model.pth
│
├── backend/
│   └── inference.py
│
├── frontend/
│   └── app.py
│
├── outputs/
│   ├── annotated_images/
│   └── prediction_logs.csv
│
├── README.md
└── requirements.txt
```

---

# 🚀 Milestone-wise Implementation

---

## 🟢 Milestone 1: Dataset Preparation & Image Processing (Weeks 1–2)

### Module 1: Dataset Setup & Image Subtraction

**Tasks**

* Load and inspect DeepPCB dataset
* Align template and test image pairs
* Perform image subtraction
* Apply Otsu thresholding and filtering

**Deliverables**

* Cleaned and aligned dataset
* Image subtraction scripts
* Sample defect-highlighted images

**Evaluation Criteria**

* Accurate defect mask generation
* Clear and visible defect regions

---

### Module 2: Contour Detection & ROI Extraction

**Tasks**

* Detect defect contours using OpenCV
* Extract bounding boxes
* Crop and label defect regions

**Deliverables**

* ROI extraction pipeline
* Cropped defect images
* Visualization of contours

**Evaluation Criteria**

* Accurate bounding boxes
* Precise ROI detection

---

## 🟡 Milestone 2: Model Training & Evaluation (Weeks 3–4)

### Module 3: Model Training

**Tasks**

* Build CNN / EfficientNet model
* Resize defect images to 128×128
* Train using Adam optimizer and Cross-Entropy loss

**Deliverables**

* Trained model file
* Accuracy and loss graphs
* Confusion matrix

**Evaluation Criteria**

* ≥ 95% classification accuracy
* Stable training performance

---

### Module 4: Evaluation & Prediction Testing

**Tasks**

* Test model on unseen PCB images
* Run inference pipeline
* Compare predictions with ground truth

**Deliverables**

* Annotated output images
* Final evaluation report

**Evaluation Criteria**

* Low false positives and false negatives

---

## 🔵 Milestone 3: Frontend & Backend Integration (Weeks 5–6)

### Module 5: Frontend Development

**Tasks**

* Build UI using Streamlit / Web stack
* Upload template and test images
* Display results with bounding boxes

**Deliverables**

* `app.py` frontend
* Real-time defect visualization

**Evaluation Criteria**

* Responsive and user-friendly UI

---

### Module 6: Backend Pipeline

**Tasks**

* Modularize preprocessing and inference
* Load trained model checkpoint
* Connect backend with frontend

**Deliverables**

* Complete backend inference pipeline
* Annotated image outputs

**Evaluation Criteria**

* Smooth image-to-result pipeline
* Minimal processing delay

---

## 🔴 Milestone 4: Finalization & Delivery (Weeks 7–8)

### Module 7: Testing, Optimization & Export

**Tasks**

* Add download option for results
* Export annotated images and CSV logs
* Optimize processing speed

**Deliverables**

* Final web app
* CSV prediction logs
* Exportable annotated images

**Evaluation Criteria**

* Fully working export functionality

---

### Module 8: Documentation & Presentation

**Tasks**

* Prepare README and documentation
* Write user guide
* Create demo video / presentation slides

**Deliverables**

* Final documentation PDF
* GitHub repository
* Demo walkthrough

**Evaluation Criteria**

* Clear documentation
* Presentation-ready project

---

## 📊 Evaluation Summary

| Milestone   | Focus             | Target                  |
| ----------- | ----------------- | ----------------------- |
| Milestone 1 | Image Processing  | Detect all defect areas |
| Milestone 2 | Model Performance | ≥ 95% accuracy          |
| Milestone 3 | UI Integration    | ≤ 3 sec/image           |
| Milestone 4 | Final Delivery    | Fully functional system |

---

## ▶️ How to Run the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run frontend/app.py
```

---

## ✅ Final Outcome

✔ Automated PCB defect detection
✔ High-accuracy defect classification
✔ Interactive web application
✔ Exportable results for reporting

---

If you want, I can next:

* Convert this into a **PDF README**
* Create a **GitHub description**
* Write a **project abstract**
* Prepare **final presentation slides**

Just tell me 👍
