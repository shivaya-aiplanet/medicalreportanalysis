import streamlit as st
import tempfile
import os
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.chat_models import ChatLiteLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
import io

# Page configuration
st.set_page_config(page_title="Medical Report Analysis", page_icon="🏥")

# Initialize session state
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def extract_text_from_image(uploaded_file):
    """Extract text from image using Azure AI Document Intelligence"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Initialize Azure AI Document Intelligence loader
        loader = AzureAIDocumentIntelligenceLoader(
            api_endpoint=st.secrets["AZURE_ENDPOINT"],
            api_key=st.secrets["AZURE_KEY"],
            file_path=tmp_file_path,
            api_model="prebuilt-layout",
            mode="markdown"
        )
        
        # Load and extract text
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Combine all document content
        extracted_text = "\n".join([doc.page_content for doc in documents])
        return extracted_text
        
    except Exception as e:
        st.error(f"Error in processing document: {str(e)}")
        return None

@st.cache_resource
def create_vector_store_cached(text):
    """Create vector store from extracted text with Qdrant Cloud and Azure OpenAI embeddings"""
    try:
        # Split text into smaller chunks for faster processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)
        
        # Create documents
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Initialize Azure OpenAI embeddings
        embeddings = AzureOpenAIEmbeddings(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            azure_deployment=st.secrets["AZURE_DEPLOYMENT"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=st.secrets["OPENAI_API_VERSION"],
            chunk_size=2048,
        )
        
        # Create Qdrant client for cloud
        client = QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"]
        )
        
        # Create vector store with Qdrant Cloud
        vector_store = Qdrant.from_documents(
            documents,
            embeddings,
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
            collection_name="medical_reports",
            force_recreate=True
        )
        
        return vector_store
    except Exception as e:
        st.error(f"Error setting up analysis system: {str(e)}")
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
        
        # Create retrieval QA chain with reduced k for faster processing
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
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
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pdf']
)

if uploaded_file is not None:
    # Display uploaded image (if it's an image)
    if uploaded_file.type.startswith('image/'):
        st.image(uploaded_file, caption="Uploaded Medical Report", use_container_width=True)
    else:
        st.write(f"Uploaded file: {uploaded_file.name}")
    
    # Process button
    if st.button("Process Report"):
        with st.spinner("📄 Reading your medical report..."):
            # Extract text using Azure AI Document Intelligence
            extracted_text = extract_text_from_image(uploaded_file)
            
            if extracted_text:
                st.session_state.processed_text = extracted_text
                
                # Create vector store using cached function for better performance
                with st.spinner("🧠 Preparing AI analysis system..."):
                    vector_store = create_vector_store_cached(extracted_text)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.chat_history = []  # Reset chat history for new document
                        st.success("✅ Report processed successfully! You can now ask questions below.")
                    else:
                        st.error("Failed to prepare analysis system")
            else:
                st.error("Failed to read the document")

# Display extracted text
if st.session_state.processed_text:
    with st.expander("View Extracted Text"):
        st.text_area("Extracted Text", st.session_state.processed_text, height=200)

# Analysis section
if st.session_state.vector_store:
    st.subheader("📊 Medical Report Analysis")
    
    # Predefined analysis options
    st.write("**Quick Analysis Options:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Comprehensive Analysis", use_container_width=True):
            query = "Provide a comprehensive analysis of this medical report"
            st.session_state.chat_history.append(("user", query))
            
        if st.button("⚠️ Abnormal Findings", use_container_width=True):
            query = "Are there any abnormal values or concerning findings?"
            st.session_state.chat_history.append(("user", query))
    
    with col2:
        if st.button("🔍 Key Findings", use_container_width=True):
            query = "What are the key findings in this report?"
            st.session_state.chat_history.append(("user", query))
            
        if st.button("📝 Follow-up Recommendations", use_container_width=True):
            query = "What follow-up actions are recommended?"
            st.session_state.chat_history.append(("user", query))
    
    st.markdown("---")
    
    # Chat interface
    st.write("**💬 Ask Questions About Your Report:**")
    
    # Display chat history
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            with st.container():
                st.write(f"**You:** {message}")
        else:
            with st.container():
                st.write(f"**AI Assistant:** {message}")
                st.markdown("---")
    
    # Process the latest question if there's one
    if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user":
        latest_query = st.session_state.chat_history[-1][1]
        
        with st.spinner("🤔 Analyzing your question..."):
            analysis_result = analyze_medical_report(st.session_state.vector_store, latest_query)
            
            if analysis_result:
                st.session_state.chat_history.append(("assistant", analysis_result))
                st.rerun()
            else:
                st.error("Sorry, I couldn't analyze your question. Please try again.")
    
    # Input for new questions
    user_question = st.text_input("Ask a follow-up question:", key="user_input")
    
    if st.button("Ask Question") and user_question:
        st.session_state.chat_history.append(("user", user_question))
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

# Disclaimer
st.markdown("---")
st.markdown("**⚠️ Disclaimer:** This tool is for educational purposes only. Always consult healthcare professionals for medical advice.")
