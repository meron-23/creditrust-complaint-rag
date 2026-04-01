from dotenv import load_dotenv
import streamlit as st
import os
import time
from src.config import Config
from src.preprocessing import load_complaints, preprocess_dataset
from src.indexer import ComplaintIndexer
from src.retriever import ComplaintRetriever
from src.generator import BusinessAnswerGenerator
from src.rag_pipeline import RAGPipeline
from src.query_validator import QueryValidator
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger(__name__)

# ---------------------------
# Professional Styling (No Emojis)
# ---------------------------
def apply_custom_styling():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background-color: #f8fafc;
        }
        
        .stButton>button {
            background-color: #1e293b;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        
        .stButton>button:hover {
            background-color: #334155;
            color: white;
            border: none;
        }
        
        .stTextInput>div>div>input {
            border-radius: 4px;
        }
        
        .sidebar .sidebar-content {
            background-color: #0f172a;
            color: white;
        }
        
        h1, h2, h3 {
            color: #0f172a;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .report-header {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 8px;
            border-bottom: 4px solid #1e293b;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .source-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 6px;
            border-left: 4px solid #94a3b8;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        .metric-label {
            color: #64748b;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metric-value {
            color: #0f172a;
            font-size: 1.5rem;
            font-weight: 700;
        }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Streamlit Setup
# ---------------------------
st.set_page_config(
    page_title="CrediTrust Financial - Complaint Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styling()

# ---------------------------
# Cached Pipeline Loading
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load and initialize the RAG pipeline with caching"""
    try:
        config = Config.from_env()
        
        # 1. Load and preprocess data
        # Note: Dynamic preprocessing happens inside the indexer/pipeline
        df = load_complaints(config.DATA_PATH)
        df_processed = preprocess_dataset(df)

        # 2. Build or load index
        indexer = ComplaintIndexer(config)

        if not os.path.exists(config.VECTOR_STORE_PATH + ".index"):
            indexer.build_index(df_processed)
            indexer.save()
        else:
            indexer.load()

        # 3. Initialize pipeline components
        retriever = ComplaintRetriever(indexer.model, indexer.index, indexer.metadatas)
        generator = BusinessAnswerGenerator(config)
        validator = QueryValidator(config)
        
        # 4. Initialize RAGPipeline
        pipeline = RAGPipeline(retriever, generator, config)
        
        return pipeline, config, validator
        
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        logger.error(f"Pipeline initialization failed: {e}")
        return None, None, None

# ---------------------------
# UI Components
# ---------------------------
def render_sidebar(_config):
    """Render the sidebar with filters and info (Strict Professional)"""
    with st.sidebar:
        st.markdown("<h2 style='color: white;'>CrediTrust</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Complaint Intelligence AI</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Parameters")
        
        selected_product = st.selectbox(
            "Product Focus:",
            ["All Products"] + _config.PRODUCTS,
            key="product_select"
        )
        
        selected_market = st.selectbox(
            "Market Focus:",
            ["All Markets"] + _config.MARKETS,
            key="market_select"
        )
        
        st.markdown("---")
        st.markdown("### Suggested Queries")
        
        example_questions = [
            "What are the top complaints about BNPL in Kenya?",
            "Analyze credit card complaint trends in Uganda",
            "What mobile app issues are customers reporting?",
            "Compare complaint patterns between Kenya and Tanzania",
            "What are emerging fraud patterns in money transfers?"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"btn_{q}"):
                st.session_state.current_question = q
                st.rerun()
        
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.current_question = ""
            st.session_state.last_answer = None
            st.session_state.last_chunks = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
            <div style='color: #94a3b8; font-size: 0.8rem;'>
            v2.0.0 (Gemini Powered)<br>
            Strict Professional Mode
            </div>
        """, unsafe_allow_html=True)
        
        # Build filters dictionary with only specified values
        filters = {}
        if selected_product != "All Products":
            filters["product"] = selected_product
        if selected_market != "All Markets":
            filters["market"] = selected_market
            
        return filters

def render_main_header():
    """Render the main header section"""
    st.markdown("""
        <div class="report-header">
            <h1>Complaint Intelligence Dashboard</h1>
            <p style="color: #64748b; max-width: 800px;">
                Strategic analysis of customer feedback narratives across East African markets. 
                Powered by Retrieval-Augmented Generation for grounded business insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_results(answer, chunks, processing_time):
    """Render the analysis results with streaming effect and professional cards"""
    st.markdown("### Strategic Analysis")
    
    # Summary of retrieval
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-label'>Source Excerpts</div><div class='metric-value'>{len(chunks)}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-label'>Response Latency</div><div class='metric-value'>{processing_time:.2f}s</div>", unsafe_allow_html=True)
    with col3:
        markets = set(c['metadata'].get('market', 'N/A') for c in chunks)
        st.markdown(f"<div class='metric-label'>Geographic Scope</div><div class='metric-value'>{', '.join(markets)}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Simulated streaming for local feel if not using direct stream
    # But since it's a generated answer, we just display it professionally
    st.markdown(answer)
    
    st.markdown("---")
    st.markdown("### Retained Evidence")
    
    for i, chunk in enumerate(chunks):
        st.markdown(f"""
            <div class="source-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 700; color: #1e293b;">Evidence Item {i+1}</span>
                    <span style="font-size: 0.8rem; color: #64748b;">ID: {chunk['metadata'].get('complaint_id', 'N/A')}</span>
                </div>
                <div style="font-size: 0.85rem; margin-bottom: 1rem; color: #334155; font-style: italic;">
                    "{chunk['text']}"
                </div>
                <div style="display: flex; gap: 1rem; font-size: 0.75rem; color: #64748b; text-transform: uppercase;">
                    <span>Product: {chunk['metadata'].get('product', 'N/A')}</span>
                    <span>Market: {chunk['metadata'].get('market', 'N/A')}</span>
                    <span>Channel: {chunk['metadata'].get('channel', 'N/A')}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Main App Logic
# ---------------------------
def main():
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ''
    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = None
    if 'last_chunks' not in st.session_state:
        st.session_state.last_chunks = None
    
    # Load pipeline
    rag_pipeline, config, validator = load_pipeline()
    
    if not rag_pipeline:
        return
    
    # Sidebar
    filters = render_sidebar(config)
    
    # Header
    render_main_header()
    
    # Question Input
    query = st.text_input(
        "Business Inquiry",
        placeholder="e.g., Identify recurring patterns in credit card billing disputes in Uganda",
        value=st.session_state.current_question,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        run_analysis = st.button("Generate Analysis", use_container_width=True)
    
    if run_analysis and query:
        # Validate query
        is_valid, validation_msg = validator.validate_query(query)
        
        if not is_valid:
            st.warning(validation_msg)
        else:
            with st.spinner("Analyzing source data..."):
                start_time = time.time()
                answer, chunks = rag_pipeline.run(query, config.TOP_K_RETRIEVAL, filters)
                duration = time.time() - start_time
                
                st.session_state.last_answer = answer
                st.session_state.last_chunks = chunks
                st.session_state.last_duration = duration
                
                render_results(answer, chunks, duration)
    
    elif st.session_state.last_answer:
        render_results(
            st.session_state.last_answer, 
            st.session_state.last_chunks, 
            st.session_state.last_duration
        )

if __name__ == "__main__":
    main()
# Run the app
# ---------------------------