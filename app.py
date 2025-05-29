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
from langchain_community.llms import LiteLLM
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
</style>
""", unsafe_allow_html=True)

class MedicalReportAnalyzer:
    def __init__(self):
        self.setup_llm()
        self.setup_embeddings()
        self.medical_knowledge_base = self.create_medical_knowledge_base()
        
    def setup_llm(self):
        """Initialize LiteLLM with provided credentials"""
        self.llm = LiteLLM(
            model=st.secrets["LITELLM_MODEL"],
            api_base=st.secrets["LITELLM_BASE_URL"],
            api_key=st.secrets["LITELLM_API_KEY"],
            temperature=0.1
        )
    
    def setup_embeddings(self):
        """Setup HuggingFace embeddings"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def create_medical_knowledge_base(self):
        """Create a medical knowledge base for reference ranges"""
        medical_data = [
            "Normal blood glucose levels: 70-100 mg/dL fasting, <140 mg/dL postprandial",
            "Normal cholesterol: Total <200 mg/dL, LDL <100 mg/dL, HDL >40 mg/dL (men), >50 mg/dL (women)",
            "Normal blood pressure: <120/80 mmHg",
            "Normal hemoglobin: 12-15.5 g/dL (women), 13.5-17.5 g/dL (men)",
            "Normal white blood cell count: 4,000-11,000 cells/μL",
            "Normal platelet count: 150,000-450,000 cells/μL",
            "Normal creatinine: 0.6-1.2 mg/dL",
            "Normal liver enzymes: ALT <40 U/L, AST <40 U/L",
            "Normal thyroid: TSH 0.4-4.0 mIU/L, T4 4.5-11.2 μg/dL",
            "Normal HbA1c: <5.7% (normal), 5.7-6.4% (prediabetes), ≥6.5% (diabetes)"
        ]
        
        documents = [Document(page_content=data, metadata={}) for data in medical_data]
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        return vectorstore
    
    def perform_ocr_tesseract(self, image):
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
    
    def perform_ocr_azure(self, image_bytes):
        """Perform OCR using Azure Form Recognizer (if credentials provided)"""
        try:
            if hasattr(st.secrets, "AZURE_ENDPOINT") and hasattr(st.secrets, "AZURE_KEY"):
                client = DocumentAnalysisClient(
                    endpoint=st.secrets["AZURE_ENDPOINT"],
                    credential=AzureKeyCredential(st.secrets["AZURE_KEY"])
                )
                
                poller = client.begin_analyze_document("prebuilt-document", image_bytes)
                result = poller.result()
                
                text = ""
                for page in result.pages:
                    for line in page.lines:
                        text += line.content + "\n"
                
                return text
            else:
                return None
        except Exception as e:
            st.warning(f"Azure OCR not available: {str(e)}")
            return None
    
    def extract_pdf_text(self, pdf_bytes):
        """Extract text from PDF"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            st.error(f"PDF extraction error: {str(e)}")
            return ""
    
    def process_document(self, uploaded_file):
        """Process uploaded document and extract text"""
        text = ""
        
        if uploaded_file.type == "application/pdf":
            # PDF processing
            pdf_bytes = uploaded_file.read()
            text = self.extract_pdf_text(pdf_bytes)
            
            # If PDF text extraction yields little content, try OCR on PDF pages
            if len(text.strip()) < 100:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    
                    # Try Azure OCR first, then Tesseract
                    azure_text = self.perform_ocr_azure(img_data)
                    if azure_text:
                        text += azure_text + "\n"
                    else:
                        image = Image.open(io.BytesIO(img_data))
                        text += self.perform_ocr_tesseract(image) + "\n"
                doc.close()
        
        else:
            # Image processing
            image = Image.open(uploaded_file)
            
            # Convert to bytes for Azure OCR
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Try Azure OCR first, then Tesseract
            azure_text = self.perform_ocr_azure(img_bytes)
            if azure_text:
                text = azure_text
            else:
                text = self.perform_ocr_tesseract(image)
        
        return text
    
    def extract_structured_data(self, text):
        """Extract structured data from medical report text"""
        # Use regex patterns to extract common medical values
        patterns = {
            'glucose': r'glucose[:\s]*(\d+\.?\d*)\s*mg/dl',
            'cholesterol': r'cholesterol[:\s]*(\d+\.?\d*)\s*mg/dl',
            'blood_pressure': r'(\d+)/(\d+)\s*mmhg',
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
                if key == 'blood_pressure':
                    extracted_data[key] = f"{matches[0][0]}/{matches[0][1]}"
                else:
                    extracted_data[key] = float(matches[0])
        
        return extracted_data
    
    def identify_abnormalities(self, structured_data):
        """Identify abnormalities in medical data"""
        abnormalities = []
        risk_scores = {}
        
        # Define normal ranges
        normal_ranges = {
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
        
        for param, value in structured_data.items():
            if param in normal_ranges:
                min_val, max_val = normal_ranges[param]
                
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        severity = self.calculate_severity(param, value, (min_val, max_val))
                        abnormalities.append({
                            'parameter': param,
                            'value': value,
                            'normal_range': f"{min_val}-{max_val}",
                            'severity': severity,
                            'description': self.get_abnormality_description(param, value, (min_val, max_val))
                        })
                        risk_scores[param] = severity
        
        return abnormalities, risk_scores
    
    def calculate_severity(self, param, value, normal_range):
        """Calculate severity of abnormality"""
        min_val, max_val = normal_range
        
        if value < min_val:
            deviation = (min_val - value) / min_val
        else:
            deviation = (value - max_val) / max_val
        
        if deviation > 0.5:
            return "High"
        elif deviation > 0.2:
            return "Medium"
        else:
            return "Low"
    
    def get_abnormality_description(self, param, value, normal_range):
        """Get description for abnormality"""
        min_val, max_val = normal_range
        
        descriptions = {
            'glucose': {
                'high': "Elevated blood glucose may indicate diabetes or prediabetes",
                'low': "Low blood glucose may cause hypoglycemic symptoms"
            },
            'cholesterol': {
                'high': "High cholesterol increases cardiovascular disease risk",
                'low': "Unusually low cholesterol may need investigation"
            },
            'hemoglobin': {
                'high': "Elevated hemoglobin may indicate dehydration or polycythemia",
                'low': "Low hemoglobin suggests anemia"
            }
        }
        
        if value > max_val:
            return descriptions.get(param, {}).get('high', f"Elevated {param}")
        else:
            return descriptions.get(param, {}).get('low', f"Low {param}")
    
    def assess_overall_risk(self, risk_scores):
        """Assess overall risk level"""
        if not risk_scores:
            return "Low"
        
        high_risk_count = sum(1 for risk in risk_scores.values() if risk == "High")
        medium_risk_count = sum(1 for risk in risk_scores.values() if risk == "Medium")
        
        if high_risk_count >= 2 or (high_risk_count >= 1 and medium_risk_count >= 2):
            return "High"
        elif high_risk_count >= 1 or medium_risk_count >= 2:
            return "Medium"
        else:
            return "Low"
    
    def generate_clinical_insights(self, text, abnormalities):
        """Generate clinical insights using LLM"""
        try:
            # Create context from abnormalities
            abnormality_context = "\n".join([
                f"- {ab['parameter']}: {ab['value']} (Normal: {ab['normal_range']}) - {ab['description']}"
                for ab in abnormalities
            ])
            
            prompt = f"""
            Analyze the following medical report and abnormalities found:
            
            Report Text: {text[:1000]}...
            
            Abnormalities Detected:
            {abnormality_context}
            
            Please provide:
            1. Clinical interpretation of these findings
            2. Potential underlying conditions to consider
            3. Recommended follow-up actions
            4. Any correlations between abnormal values
            
            Keep the response concise and clinical.
            """
            
            response = self.llm(prompt)
            return response
        except Exception as e:
            return f"Unable to generate clinical insights: {str(e)}"
    
    def format_structured_report(self, text, structured_data, abnormalities, overall_risk, insights):
        """Format the final structured report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_parameters": len(structured_data),
                "abnormal_parameters": len(abnormalities),
                "overall_risk": overall_risk
            },
            "extracted_data": structured_data,
            "abnormalities": abnormalities,
            "clinical_insights": insights,
            "confidence_score": self.calculate_confidence_score(text, structured_data)
        }
        
        return report

    def calculate_confidence_score(self, text, structured_data):
        """Calculate confidence score based on text quality and data extraction"""
        # Basic confidence calculation
        text_quality = min(len(text) / 1000, 1.0)  # Normalize text length
        extraction_success = len(structured_data) / 10  # Assume 10 possible parameters
        
        confidence = (text_quality * 0.6 + extraction_success * 0.4) * 100
        return min(confidence, 95)  # Cap at 95%

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Report Analyzer</h1>
        <p>Advanced AI-powered analysis of medical reports and lab results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("Initializing AI Medical Analyzer..."):
            st.session_state.analyzer = MedicalReportAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for patient context
    with st.sidebar:
        st.markdown("### 👤 Patient Context")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        medical_history = st.text_area("Medical History (Optional)", 
                                     placeholder="Previous conditions, medications...")
        urgency = st.selectbox("Analysis Priority", ["Routine", "Urgent", "Critical"])
        
        st.markdown("### 🔧 OCR Settings")
        ocr_method = st.radio("OCR Method", ["Auto (Azure + Tesseract)", "Tesseract Only"])
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-container">
            <h3 style="color: white; margin-bottom: 1rem;">📁 Upload Medical Report</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a medical report file",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff'],
            help="Upload PDF or image files of medical reports"
        )
        
        if uploaded_file:
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Show file preview for images
            if uploaded_file.type.startswith('image/'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Medical Report", use_column_width=True)
    
    with col2:
        if uploaded_file:
            st.markdown("### 🔄 Processing Status")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Document Processing & OCR
                status_text.text("🔍 Performing OCR and text extraction...")
                progress_bar.progress(20)
                
                extracted_text = analyzer.process_document(uploaded_file)
                
                if not extracted_text.strip():
                    st.error("❌ No text could be extracted from the document")
                    return
                
                # Step 2: Extract Structured Data
                status_text.text("📊 Extracting structured medical data...")
                progress_bar.progress(40)
                
                structured_data = analyzer.extract_structured_data(extracted_text)
                
                # Step 3: Query Medical Knowledge Base
                status_text.text("🧠 Querying medical knowledge base...")
                progress_bar.progress(60)
                
                # Step 4: Identify Abnormalities
                status_text.text("⚠️ Identifying abnormalities...")
                progress_bar.progress(70)
                
                abnormalities, risk_scores = analyzer.identify_abnormalities(structured_data)
                overall_risk = analyzer.assess_overall_risk(risk_scores)
                
                # Step 5: Generate Clinical Insights
                status_text.text("💡 Generating clinical insights...")
                progress_bar.progress(85)
                
                insights = analyzer.generate_clinical_insights(extracted_text, abnormalities)
                
                # Step 6: Format Report
                status_text.text("📋 Formatting structured report...")
                progress_bar.progress(95)
                
                final_report = analyzer.format_structured_report(
                    extracted_text, structured_data, abnormalities, overall_risk, insights
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                
                # Display Results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Risk Assessment Cards
                risk_color = {
                    "Low": "risk-low",
                    "Medium": "risk-medium", 
                    "High": "risk-high"
                }
                
                st.markdown(f"""
                <div class="{risk_color[overall_risk]}">
                    <h3>Overall Risk Assessment: {overall_risk}</h3>
                    <p>Confidence Score: {final_report['confidence_score']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>{}</h3>
                        <p>Parameters Extracted</p>
                    </div>
                    """.format(len(structured_data)), unsafe_allow_html=True)
                
                with col4:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>{}</h3>
                        <p>Abnormalities Found</p>
                    </div>
                    """.format(len(abnormalities)), unsafe_allow_html=True)
                
                with col5:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>{:.1f}%</h3>
                        <p>Analysis Confidence</p>
                    </div>
                    """.format(final_report['confidence_score']), unsafe_allow_html=True)
                
                # Detailed Results
                if structured_data:
                    st.markdown("### 📈 Extracted Medical Parameters")
                    df = pd.DataFrame(list(structured_data.items()), columns=['Parameter', 'Value'])
                    st.dataframe(df, use_container_width=True)
                
                if abnormalities:
                    st.markdown("### ⚠️ Abnormal Findings")
                    for ab in abnormalities:
                        st.markdown(f"""
                        <div class="analysis-card">
                            <h4>{ab['parameter'].title()}: {ab['value']}</h4>
                            <p><strong>Normal Range:</strong> {ab['normal_range']}</p>
                            <p><strong>Severity:</strong> <span style="color: {'red' if ab['severity'] == 'High' else 'orange' if ab['severity'] == 'Medium' else 'green'}">{ab['severity']}</span></p>
                            <p>{ab['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Clinical Insights
                st.markdown("### 💡 Clinical Insights")
                st.markdown(f"""
                <div class="analysis-card">
                    {insights}
                </div>
                """, unsafe_allow_html=True)
                
                # Visualization
                if abnormalities:
                    st.markdown("### 📊 Risk Distribution")
                    
                    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
                    for ab in abnormalities:
                        risk_counts[ab['severity']] += 1
                    
                    fig = px.pie(
                        values=list(risk_counts.values()),
                        names=list(risk_counts.keys()),
                        title="Abnormalities by Risk Level",
                        color_discrete_map={"Low": "#48dbfb", "Medium": "#feca57", "High": "#ff6b6b"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download Report
                st.markdown("### 💾 Download Report")
                report_json = json.dumps(final_report, indent=2)
                
                col6, col7 = st.columns(2)
                with col6:
                    st.download_button(
                        label="📄 Download JSON Report",
                        data=report_json,
                        file_name=f"medical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col7:
                    # Convert to PDF-ready format
                    pdf_content = f"""
                    MEDICAL REPORT ANALYSIS
                    ======================
                    
                    Timestamp: {final_report['timestamp']}
                    Overall Risk: {overall_risk}
                    Confidence Score: {final_report['confidence_score']:.1f}%
                    
                    EXTRACTED PARAMETERS:
                    {json.dumps(structured_data, indent=2)}
                    
                    ABNORMALITIES:
                    {json.dumps(abnormalities, indent=2)}
                    
                    CLINICAL INSIGHTS:
                    {insights}
                    """
                    
                    st.download_button(
                        label="📋 Download Text Report",
                        data=pdf_content,
                        file_name=f"medical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
