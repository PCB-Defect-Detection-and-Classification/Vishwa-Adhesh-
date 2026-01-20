import streamlit as st
from PIL import Image
import cv2
import numpy as np
from backend import detect_defect

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="PCB Defect Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM STYLING =====
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { 
        background-color: #FF6B6B;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        width: 100%;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.title("🔬 PCB Defect Detection System")
st.markdown("**Intelligent inspection for printed circuit boards**")
st.divider()

# ===== SIDEBAR =====
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This system detects defects in PCB images using:
    - Image Processing (Thresholding)
    - Morphological Operations
    - Contour Detection
    
    **Supported Formats:** JPG, PNG, JPEG
    """)
    
    st.divider()
    st.write("**Infosys Springboard Project**")
    st.write("Milestone 3: Frontend + Backend Integration")

# ===== MAIN CONTENT =====
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload PCB Image")
    
    uploaded_file = st.file_uploader(
        "Choose a PCB image",
        type=["jpg", "png", "jpeg"],
        help="Upload a clear PCB image for defect detection"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("📊 Detection Results")
    
    if uploaded_file is not None:
        if st.button("🔍 ANALYZE PCB", key="detect_btn"):
            with st.spinner("🔄 Analyzing image..."):
                # Run detection
                result_img, defect_info = detect_defect(image)
                
                # Display result image
                st.image(result_img, caption="Detection Result", use_column_width=True)
                
                # Show detailed results
                st.divider()
                
                if defect_info['count'] > 0:
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>✅ DEFECT DETECTED</h4>
                    <p><strong>Total Defects:</strong> {defect_info['count']}</p>
                    <p><strong>Confidence:</strong> {defect_info['confidence']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show individual defects
                    st.write("### Defect Details:")
                    for i, defect in enumerate(defect_info['defects'], 1):
                        st.write(f"""
                        **Defect #{i}**
                        - Position: ({defect['x']}, {defect['y']})
                        - Size: {defect['width']}×{defect['height']} pixels
                        - Area: {int(defect['area'])} pixels²
                        """)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h4>✅ NO DEFECT FOUND</h4>
                    <p>PCB appears to be in good condition</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                st.success("Detection completed successfully!")
    else:
        st.info("👆 Upload an image to start analysis")

# ===== FOOTER =====
st.divider()
st.markdown("""
---
**System Info:**
- Detection Method: Morphological Image Processing
- Confidence Level: High Accuracy
- Processing Time: Real-time
- Status: ✅ Active
""")

