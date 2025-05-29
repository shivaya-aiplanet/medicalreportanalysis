# Required packages:
# streamlit>=1.24.0
# pandas>=1.5.0
# numpy>=1.21.0
# Pillow>=9.0.0
# plotly>=5.13.0
# langchain>=0.1.0
# faiss-cpu>=1.7.4
# sentence-transformers>=2.2.2
# pytesseract>=0.3.10
# opencv-python>=4.7.0
# PyMuPDF>=1.21.0
# azure-ai-formrecognizer>=3.2.0
# litellm>=1.0.0
# langchain-community>=0.0.10

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
import json
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from litellm import completion
from langchain.schema import Document

# OCR and document processing
import pytesseract
import cv2
import fitz  # PyMuPDF
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

# Set page config for wide layout and custom theme
st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .upload-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

class MedicalReportAnalyzer:
    def __init__(self):
        self.normal_ranges = {
            'glucose': (70, 100),
            'cholesterol': (0, 200),
            'hemoglobin': (12, 17.5),
            'wbc': (4000, 11000),
            'creatinine': (0.6, 1.2),
            'alt': (0, 40),
            'ast': (0, 40),
            'tsh': (0.4, 4.0),
            'hba1c': (0, 5.7)
        }
    
    def perform_ocr(self, image):
        """Perform OCR using Tesseract"""
        try:
            # Convert PIL image to OpenCV format
            img_array = np.array(image)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Perform OCR
            text = pytesseract.image_to_string(gray, config='--psm 6')
            return text
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return ""
    
    def extract_medical_values(self, text):
        """Extract medical values from text"""
        patterns = {
            'glucose': r'glucose[:\s]*(\d+\.?\d*)\s*mg/dl',
            'cholesterol': r'cholesterol[:\s]*(\d+\.?\d*)\s*mg/dl',
            'hemoglobin': r'hemoglobin[:\s]*(\d+\.?\d*)\s*g/dl',
            'wbc': r'wbc[:\s]*(\d+\.?\d*)',
            'creatinine': r'creatinine[:\s]*(\d+\.?\d*)\s*mg/dl',
            'alt': r'alt[:\s]*(\d+\.?\d*)\s*u/l',
            'ast': r'ast[:\s]*(\d+\.?\d*)\s*u/l',
            'tsh': r'tsh[:\s]*(\d+\.?\d*)',
            'hba1c': r'hba1c[:\s]*(\d+\.?\d*)%?'
        }
        
        extracted_data = {}
        text_lower = text.lower()
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                extracted_data[key] = float(matches[0])
        
        return extracted_data
    
    def analyze_values(self, values):
        """Analyze medical values and identify abnormalities"""
        abnormalities = []
        
        for param, value in values.items():
            if param in self.normal_ranges:
                min_val, max_val = self.normal_ranges[param]
                if value < min_val or value > max_val:
                    status = "High" if value > max_val else "Low"
                    abnormalities.append({
                        'parameter': param,
                        'value': value,
                        'normal_range': f"{min_val}-{max_val}",
                        'status': status
                    })
        
        return abnormalities

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Report Analyzer</h1>
        <p>Simple OCR-based analysis of medical reports</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = MedicalReportAnalyzer()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Medical Report Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of your medical report"
    )
    
    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Report", use_column_width=True)
        
        # Process image
        with st.spinner("Analyzing medical report..."):
            # Perform OCR
            text = analyzer.perform_ocr(image)
            
            if text:
                # Extract values
                values = analyzer.extract_medical_values(text)
                
                # Analyze values
                abnormalities = analyzer.analyze_values(values)
                
                # Display results
                st.markdown("### 📊 Analysis Results")
                
                # Show extracted values
                if values:
                    st.markdown("#### Extracted Values")
                    df = pd.DataFrame(list(values.items()), columns=['Parameter', 'Value'])
                    st.dataframe(df, use_container_width=True)
                
                # Show abnormalities
                if abnormalities:
                    st.markdown("#### ⚠️ Abnormal Values")
                    for ab in abnormalities:
                        st.markdown(f"""
                        <div class="result-card">
                            <h4>{ab['parameter'].title()}: {ab['value']}</h4>
                            <p><strong>Normal Range:</strong> {ab['normal_range']}</p>
                            <p><strong>Status:</strong> {ab['status']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No abnormalities detected in the extracted values.")
                
                # Show raw OCR text
                with st.expander("View Raw OCR Text"):
                    st.text(text)
            else:
                st.error("❌ Could not extract text from the image. Please try a clearer image.")

if __name__ == "__main__":
    main()
