import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from litellm import completion
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

# Function to get LLM response
def get_llm_response(prompt, context):
    try:
        response = completion(
            model=os.getenv("LITELLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ],
            api_key=os.getenv("LITELLM_API_KEY", "sk-V12plNmxne0F7XIQuyzJDQ"),
            base_url=os.getenv("LITELLM_BASE_URL", "https://litellm.aiplanet.com/"),
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting LLM response: {str(e)}")
        return None

# Main app
def main():
    # File uploader
    uploaded_file = st.file_uploader("Upload Medical Report", type=['pdf', 'png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Process document
        chunks = process_document(uploaded_file)
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector store
        vectorstore = Qdrant.from_texts(
            chunks,
            embeddings,
            location=":memory:"
        )
        
        # Chat interface
        st.subheader("Ask questions about the report")
        user_question = st.text_input("Your question:")
        
        if user_question:
            with st.spinner("Analyzing..."):
                # Get relevant context
                docs = vectorstore.similarity_search(user_question, k=3)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Get response from LLM
                response = get_llm_response(user_question, context)
                
                if response:
                    # Update chat history
                    st.session_state.chat_history.append((user_question, response))
                    
                    # Display response
                    st.write("Answer:", response)
                    
                    # Display sources
                    with st.expander("View Sources"):
                        for doc in docs:
                            st.write(doc.page_content)

if __name__ == "__main__":
    main()
