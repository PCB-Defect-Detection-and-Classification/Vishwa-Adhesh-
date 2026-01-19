Milestone 1: Dataset Preparation & Image Processing
Overview:

  This milestone focuses on preparing the DeepPCB dataset and applying image subtraction–based defect detection. Template and test PCB images are aligned, subtracted, and processed to generate accurate defect masks.

Tasks Completed:

   Dataset setup and inspection (DeepPCB)

   Image alignment and preprocessing

   Image subtraction for defect difference maps

   Otsu’s thresholding for defect segmentation

   Noise removal using morphological filtering

Deliverables:

   Cleaned and aligned dataset

   Image subtraction & thresholding script

   Sample defect-highlighted images

Project Structure:
Milestone-1/
├── dataset/
├── scripts/
├── outputs/
├── README.md
└── requirements.txt

Evaluation Focus:

   Accurate defect mask generation

  Proper image alignment

  Clear subtraction results

Technologies Used:

  Python

  OpenCV

  NumPy

How to Run:
  pip install -r requirements.txt
  python scripts/subtraction_thresholding.py
