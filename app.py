import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LiteLLM
import pytesseract
from PIL import Image
import io
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="Medical Report Analysis", layout="wide")
st.title("Medical Report Analysis")

# System prompt for medical analysis
SYSTEM_PROMPT = """You are a medical report analysis assistant. Your role is to:
1. Analyze medical reports and provide clear, accurate interpretations
2. Highlight important medical findings and their implications
3. Explain medical terminology in simple terms when needed
4. Maintain professional and empathetic communication
5. Focus on factual information from the report
6. Avoid making definitive diagnoses or treatment recommendations
7. Always cite specific parts of the report when providing information

Remember to:
- Be precise and clear in your explanations
- Use appropriate medical terminology while remaining accessible
- Maintain patient confidentiality
- Focus on the information present in the report
- Indicate when information is unclear or missing"""

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to process uploaded document
def process_document(uploaded_file):
    with st.spinner("Processing document..."):
        # Read the file
        file_bytes = uploaded_file.read()
        
        # Convert to image for OCR
        image = Image.open(io.BytesIO(file_bytes))
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        return chunks

# Function to initialize the QA chain
def initialize_qa_chain():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize LiteLLM with specific configuration
    llm = LiteLLM(
        api_key=os.getenv("LITELLM_API_KEY", "sk-V12plNmxne0F7XIQuyzJDQ"),
        model_name=os.getenv("LITELLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("LITELLM_BASE_URL", "https://litellm.aiplanet.com/"),
        temperature=0.3,
        system_prompt=SYSTEM_PROMPT
    )
    
    return embeddings, llm

# Main app
def main():
    # File uploader
    uploaded_file = st.file_uploader("Upload Medical Report", type=['pdf', 'png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Process document
        chunks = process_document(uploaded_file)
        
        # Initialize QA chain
        embeddings, llm = initialize_qa_chain()
        
        # Create vector store
        vectorstore = Qdrant.from_texts(
            chunks,
            embeddings,
            location=":memory:"
        )
        
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        # Chat interface
        st.subheader("Ask questions about the report")
        user_question = st.text_input("Your question:")
        
        if user_question:
            with st.spinner("Analyzing..."):
                # Get response
                response = qa_chain({"question": user_question, "chat_history": st.session_state.chat_history})
                
                # Update chat history
                st.session_state.chat_history.append((user_question, response["answer"]))
                
                # Display response
                st.write("Answer:", response["answer"])
                
                # Display sources
                with st.expander("View Sources"):
                    for doc in response["source_documents"]:
                        st.write(doc.page_content)

if __name__ == "__main__":
    main()
