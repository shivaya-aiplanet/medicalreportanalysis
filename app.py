import streamlit as st
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from langchain_community.chat_models import ChatLiteLLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Medical Q&A Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

def setup_llm():
    """Setup LiteLLM with API credentials"""
    return ChatLiteLLM(
        model=st.secrets["LITELLM_MODEL"],
        api_base=st.secrets["LITELLM_BASE_URL"],
        api_key=st.secrets["LITELLM_API_KEY"],
        temperature=0.1
    )

def setup_embeddings():
    """Setup HuggingFace embeddings"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_data(ttl=3600)  # Cache for 1 hour
def search_medical_web(query, max_results=5):
    """Search web for medical information with caching"""
    try:
        # Initialize DuckDuckGo search
        search = DuckDuckGoSearchAPIWrapper(max_results=max_results)
        
        # Add medical context to search query
        medical_query = f"medical health {query} site:mayoclinic.org OR site:webmd.org OR site:healthline.com OR site:medlineplus.gov OR site:nih.gov"
        
        # Perform search
        search_results = search.run(medical_query)
        return search_results
        
    except Exception as e:
        st.warning(f"Web search temporarily unavailable: {str(e)}")
        return None

def extract_urls_from_search(search_results):
    """Extract URLs from search results"""
    urls = []
    if search_results:
        # Simple URL extraction from DuckDuckGo results
        lines = search_results.split('\n')
        for line in lines:
            if 'http' in line:
                # Extract URL from the line
                start = line.find('http')
                if start != -1:
                    end = line.find(' ', start)
                    if end == -1:
                        end = len(line)
                    url = line[start:end].strip('.,)]')
                    if url.endswith(('mayoclinic.org', 'webmd.com', 'healthline.com', 'medlineplus.gov', 'nih.gov')):
                        urls.append(url)
    return urls[:3]  # Limit to 3 URLs for speed

async def fetch_web_content(urls):
    """Asynchronously fetch content from medical websites"""
    try:
        # Load HTML content asynchronously
        loader = AsyncHtmlLoader(urls)
        docs = await asyncio.to_thread(loader.load)
        
        # Transform HTML to clean text
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            docs, 
            tags_to_extract=["p", "h1", "h2", "h3", "li"]
        )
        
        # Filter and clean content
        medical_docs = []
        for doc in docs_transformed:
            if len(doc.page_content) > 100:  # Only keep substantial content
                # Clean and truncate content
                content = doc.page_content[:1500]  # Limit content length
                medical_docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": doc.metadata.get("source", "web"),
                        "type": "web_search",
                        "timestamp": datetime.now().isoformat()
                    }
                ))
        
        return medical_docs
        
    except Exception as e:
        st.warning(f"Content extraction failed: {str(e)}")
        return []

def create_base_medical_knowledge():
    """Create base medical knowledge for offline capability"""
    base_knowledge = [
        "Emergency symptoms requiring immediate medical attention include chest pain, difficulty breathing, severe bleeding, loss of consciousness, and signs of stroke.",
        "Common vital signs include body temperature (98.6°F/37°C normal), blood pressure (120/80 mmHg normal), heart rate (60-100 bpm normal), and respiratory rate (12-20 breaths/min normal).",
        "Medication safety guidelines: Always follow prescribed dosages, check for drug interactions, store medications properly, and consult healthcare providers before stopping treatments.",
        "Preventive care recommendations include regular check-ups, vaccinations, cancer screenings, and lifestyle modifications for chronic disease prevention.",
        "Mental health warning signs include persistent sadness, anxiety, changes in sleep or appetite, social withdrawal, and thoughts of self-harm requiring professional help."
    ]
    
    return [Document(
        page_content=content, 
        metadata={"source": "base_knowledge", "type": "offline"}
    ) for content in base_knowledge]

def setup_hybrid_vector_store(query=None, enable_web_search=True):
    """Setup vector store with both base knowledge and web search results"""
    with st.spinner("🔄 Preparing medical knowledge base..."):
        try:
            embeddings = setup_embeddings()
            
            # Start with base medical knowledge
            documents = create_base_medical_knowledge()
            
            # Add web search results if enabled and query provided
            if enable_web_search and query:
                with st.spinner("🌐 Searching medical websites..."):
                    # Search web for current medical information
                    search_results = search_medical_web(query)
                    
                    if search_results:
                        # Extract URLs from search results
                        urls = extract_urls_from_search(search_results)
                        
                        if urls:
                            # Fetch web content asynchronously
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                web_docs = loop.run_until_complete(fetch_web_content(urls))
                                loop.close()
                                
                                if web_docs:
                                    documents.extend(web_docs)
                                    st.success(f"✅ Retrieved {len(web_docs)} medical web sources")
                                else:
                                    st.info("📚 Using offline medical knowledge")
                            except Exception as e:
                                st.warning("🔄 Using offline medical knowledge")
                        else:
                            st.info("📚 Using offline medical knowledge")
            
            # Split documents for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            
            # Setup Qdrant client (in-memory for speed)
            client = QdrantClient(":memory:")
            
            # Create collection
            client.create_collection(
                collection_name="medical_knowledge",
                vectors_config=models.VectorParams(
                    size=384,  # MiniLM embedding size
                    distance=models.Distance.COSINE
                )
            )
            
            # Create vector store
            vector_store = Qdrant(
                client=client,
                collection_name="medical_knowledge",
                embeddings=embeddings
            )
            
            # Add documents
            vector_store.add_documents(splits)
            
            return vector_store, len([d for d in documents if d.metadata.get("type") == "web_search"])
            
        except Exception as e:
            st.error(f"Error setting up knowledge base: {str(e)}")
            return None, 0

def create_hybrid_qa_chain(vector_store, user_context, has_web_data=False):
    """Create QA chain that combines LLM reasoning with retrieved knowledge"""
    llm = setup_llm()
    
    # Enhanced prompt template for hybrid reasoning
    if user_context["is_professional"]:
        template = """You are an advanced medical AI assistant providing evidence-based information to healthcare professionals.

Available Context: {context}

Medical Query: {question}

Instructions:
1. Analyze the provided medical context from both clinical knowledge and current sources
2. Provide detailed clinical information including differential diagnoses where relevant
3. Include current medical guidelines and evidence-based recommendations
4. Note any limitations in the available information
5. Indicate confidence level and suggest additional clinical considerations

""" + ("Note: This response includes current web-based medical information." if has_web_data else "Note: This response is based on general medical knowledge.") + """

IMPORTANT: This information supports but does not replace clinical judgment and direct patient assessment.

Clinical Response:"""
    else:
        template = """You are a helpful medical AI assistant providing reliable health information to patients and the general public.

Available Information: {context}

Health Question: {question}

Instructions:
1. Provide a clear, easy-to-understand explanation
2. Include practical guidance and when to seek medical care
3. Emphasize important safety considerations
4. Be supportive while maintaining medical accuracy
5. Indicate confidence level in the information provided

""" + ("Note: This includes current information from trusted medical websites." if has_web_data else "Note: This is based on general medical knowledge.") + """

CRITICAL DISCLAIMER: This information is for educational purposes only and does not constitute medical advice. Always consult qualified healthcare professionals for proper diagnosis, treatment, and medical decisions.

Response:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def generate_hybrid_response(question, user_context, enable_web_search=True):
    """Generate response using both LLM reasoning and web search"""
    start_time = time.time()
    
    # Setup hybrid vector store
    vector_store, web_sources_count = setup_hybrid_vector_store(
        query=question if enable_web_search else None,
        enable_web_search=enable_web_search
    )
    
    if vector_store is None:
        return None, 0, 0
    
    # Create QA chain
    with st.spinner("🧠 Analyzing medical information..."):
        qa_chain = create_hybrid_qa_chain(vector_store, user_context, web_sources_count > 0)
        
        try:
            result = qa_chain({"query": question})
            
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000)  # Convert to milliseconds
            
            return result, web_sources_count, response_time
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None, 0, 0

def main():
    # Header
    st.title("🏥 Medical Q&A Assistant")
    st.markdown("*Combining AI reasoning with real-time medical web search*")
    
    # Sidebar for user context and settings
    with st.sidebar:
        st.header("👤 User Information")
        
        user_type = st.selectbox(
            "I am a:",
            ["Patient/General Public", "Healthcare Professional"],
            help="This helps tailor the response appropriately"
        )
        
        urgency = st.selectbox(
            "Urgency Level:",
            ["General Information", "Moderate Concern", "High Priority"],
            help="Indicates the urgency of your medical question"
        )
        
        st.markdown("---")
        st.header("⚙️ Search Settings")
        
        enable_web_search = st.toggle(
            "🌐 Enable Web Search",
            value=True,
            help="Include current medical information from trusted websites"
        )
        
        if enable_web_search:
            st.success("✅ Web search enabled")
            st.caption("Sources: Mayo Clinic, WebMD, Healthline, MedlinePlus, NIH")
        else:
            st.info("📚 Using offline knowledge only")
        
        st.markdown("---")
        st.markdown("### ⚠️ Important Notice")
        st.error(
            "🚨 **EMERGENCY:** For medical emergencies, call emergency services immediately. "
            "This AI assistant provides general information only and does not replace professional medical care."
        )
    
    # Main interface
    st.markdown("### 💬 Ask Your Medical Question")
    
    question = st.text_area(
        "Enter your medical question:",
        placeholder="e.g., What are the latest treatments for type 2 diabetes?",
        height=100
    )
    
    if st.button("🔍 Get Medical Information", type="primary"):
        if not question.strip():
            st.warning("Please enter a medical question.")
            return
        
        # Classify user context
        user_context = {
            "user_type": user_type,
            "urgency": urgency,
            "is_professional": user_type == "Healthcare Professional"
        }
        
        # Generate hybrid response
        result, web_sources, response_time = generate_hybrid_response(
            question, user_context, enable_web_search
        )
        
        if result:
            # Display response
            st.markdown("### 📋 Medical Information")
            
            # Response metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⚡ Response Time", f"{response_time}ms")
            with col2:
                st.metric("🌐 Web Sources", web_sources)
            with col3:
                st.metric("📊 Total Sources", len(result.get('source_documents', [])))
            
            # Main answer
            st.markdown("**Medical Response:**")
            st.write(result['result'])
            
            # Source analysis
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Information Quality:**")
                if web_sources > 0:
                    st.success(f"✅ Current information from {web_sources} trusted medical websites")
                else:
                    st.info("📚 Based on general medical knowledge")
                
            with col2:
                st.markdown("**👤 Response Type:**")
                st.info(f"Tailored for: {user_type}")
            
            # Detailed sources
            if result.get('source_documents'):
                with st.expander("📚 Knowledge Sources & Evidence"):
                    web_sources_found = []
                    offline_sources_found = []
                    
                    for i, doc in enumerate(result['source_documents']):
                        source_type = doc.metadata.get('type', 'unknown')
                        if source_type == 'web_search':
                            web_sources_found.append(doc)
                        else:
                            offline_sources_found.append(doc)
                    
                    if web_sources_found:
                        st.markdown("**🌐 Current Web Sources:**")
                        for i, doc in enumerate(web_sources_found):
                            st.write(f"**Source {i+1}:** {doc.page_content[:300]}...")
                            if 'source' in doc.metadata:
                                st.caption(f"From: {doc.metadata['source']}")
                    
                    if offline_sources_found:
                        st.markdown("**📚 Medical Knowledge Base:**")
                        for i, doc in enumerate(offline_sources_found):
                            st.write(f"**Reference {i+1}:** {doc.page_content[:200]}...")
            
            # Related actions
            st.markdown("---")
            st.markdown("**🔗 Recommended Next Steps:**")
            
            action_cols = st.columns(4)
            with action_cols[0]:
                if st.button("🩺 Symptoms", key="symptoms"):
                    st.info("Consider asking about specific symptoms related to your condition")
            with action_cols[1]:
                if st.button("💊 Treatment", key="treatment"):
                    st.info("Ask about treatment options and management strategies")
            with action_cols[2]:
                if st.button("🔬 Diagnosis", key="diagnosis"):
                    st.info("Inquire about diagnostic procedures and tests")
            with action_cols[3]:
                if st.button("🏥 When to See Doctor", key="doctor"):
                    st.info("Ask when professional medical consultation is needed")
            
            # Final comprehensive disclaimer
            st.markdown("---")
            st.error(
                "⚠️ **COMPREHENSIVE MEDICAL DISCLAIMER:** This AI assistant provides general medical information for educational purposes only. "
                "It does not constitute professional medical advice, diagnosis, or treatment recommendations. "
                "The information may not reflect the most current medical developments. "
                "Always consult qualified healthcare professionals for medical concerns, treatment decisions, and health management. "
                "In case of medical emergencies, contact emergency services immediately."
            )

if __name__ == "__main__":
    main()
