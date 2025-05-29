import streamlit as st
import pytesseract
from PIL import Image
import tempfile
import os
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.chat_models import ChatLiteLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import io

# Page configuration
st.set_page_config(page_title="Medical Report Analysis", page_icon="🏥")

# Initialize session state
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def extract_text_from_image(image):
    """Extract text from image using Tesseract OCR"""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error in OCR processing: {str(e)}")
        return None

def create_vector_store(text):
    """Create vector store from extracted text"""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # Create documents
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vector_store = Qdrant.from_documents(
            documents,
            embeddings,
            location=":memory:",
            collection_name="medical_reports"
        )
        
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def analyze_medical_report(vector_store, query):
    """Analyze medical report using LiteLLM"""
    try:
        # Initialize LiteLLM
        llm = ChatLiteLLM(
            model=st.secrets["LITELLM_MODEL"],
            api_key=st.secrets["LITELLM_API_KEY"],
            api_base=st.secrets["LITELLM_BASE_URL"]
        )
        
        # Create prompt template
        prompt_template = """
        You are a medical AI assistant analyzing a medical report. Based on the following context from the medical report, please provide a comprehensive analysis.

        Context: {context}

        Question: {question}

        Please provide a detailed analysis focusing on:
        1. Key medical findings
        2. Potential concerns or abnormalities
        3. Recommendations for follow-up
        4. Overall assessment

        Analysis:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        # Get response
        response = qa_chain.run(query)
        return response
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return None

# Main app
st.title("🏥 Medical Report Analysis")
st.write("Upload a medical report image for AI-powered analysis")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a medical report image",
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Medical Report", use_column_width=True)
    
    # Process button
    if st.button("Process Report"):
        with st.spinner("Extracting text from image...", show_time=True):
            # Extract text using OCR
            extracted_text = extract_text_from_image(image)
            
            if extracted_text:
                st.session_state.processed_text = extracted_text
                
                # Create vector store
                with st.spinner("Creating knowledge base...", show_time=True):
                    vector_store = create_vector_store(extracted_text)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.success("Report processed successfully!")
                    else:
                        st.error("Failed to create knowledge base")
            else:
                st.error("Failed to extract text from image")

# Display extracted text
if st.session_state.processed_text:
    with st.expander("View Extracted Text"):
        st.text_area("Extracted Text", st.session_state.processed_text, height=200)

# Analysis section
if st.session_state.vector_store:
    st.subheader("📊 Medical Report Analysis")
    
    # Predefined analysis options
    analysis_options = [
        "Provide a comprehensive analysis of this medical report",
        "What are the key findings in this report?",
        "Are there any abnormal values or concerning findings?",
        "What follow-up actions are recommended?",
        "Summarize the patient's condition"
    ]
    
    selected_analysis = st.selectbox("Choose analysis type:", analysis_options)
    
    # Custom query option
    custom_query = st.text_input("Or ask a custom question about the report:")
    
    if st.button("Analyze Report"):
        query = custom_query if custom_query else selected_analysis
        
        with st.spinner("Analyzing medical report...", show_time=True):
            analysis_result = analyze_medical_report(st.session_state.vector_store, query)
            
            if analysis_result:
                st.subheader("🔍 Analysis Results")
                st.write(analysis_result)
            else:
                st.error("Failed to analyze the report")

# Sidebar with information
st.sidebar.title("ℹ️ About")
st.sidebar.write("""
This app uses:
- **Tesseract OCR** for text extraction
- **LangChain** for document processing
- **Qdrant** for vector storage
- **HuggingFace** for embeddings
- **LiteLLM** for AI analysis

**Note:** This is for educational purposes only and should not replace professional medical advice.
""")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This tool is for educational purposes only. Always consult healthcare professionals for medical advice.")
