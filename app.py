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

# Setup logger
logger = setup_logger(__name__)

# ---------------------------
# Streamlit Setup
# ---------------------------
st.set_page_config(
    page_title="CrediTrust Financial - Complaint Insights AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Cached Pipeline Loading
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load and initialize the RAG pipeline with caching"""
    try:
        config = Config.from_env()
        
        # 1. Load and preprocess data
        with st.spinner("📊 Loading complaint data..."):
            df = load_complaints(config.DATA_PATH)
            df = preprocess_dataset(df)

        # 2. Build or load index
        indexer = ComplaintIndexer(config)

        if not os.path.exists(config.VECTOR_STORE_PATH + ".index"):
            with st.spinner("🔨 Building search index..."):
                indexer.build_index(df)
                indexer.save()
        else:
            with st.spinner("📂 Loading existing index..."):
                indexer.load()

        # 3. Initialize pipeline components
        retriever = ComplaintRetriever(indexer.model, indexer.index, indexer.metadatas)
        generator = BusinessAnswerGenerator(config)
        validator = QueryValidator(config)
        
        return RAGPipeline(retriever, generator), config, validator
        
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {str(e)}")
        return None, None, None

# ---------------------------
# UI Components
# ---------------------------
def render_sidebar():
    """Render the sidebar with filters and info"""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/0066cc/ffffff?text=CrediTrust", width=150)
        st.title("🤖 Complaint Insights AI")
        
        st.markdown("---")
        st.markdown("### 📊 Filters")
        
        # Product filter
        selected_product = st.selectbox(
            "Filter by Product:",
            ["All Products"] + config.PRODUCTS,
            help="Focus analysis on specific products"
        )
        
        # Market filter
        selected_market = st.selectbox(
            "Filter by Market:",
            ["All Markets"] + config.MARKETS,
            help="Focus analysis on specific regions"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        
        example_questions = [
            "What are the top complaints about BNPL in Kenya?",
            "Analyze credit card complaint trends in Uganda",
            "What mobile app issues are customers reporting?",
            "Compare complaint patterns between Kenya and Tanzania",
            "What are emerging fraud patterns in money transfers?"
        ]
        
        for question in example_questions:
            if st.button(f"• {question}", key=question):
                st.session_state.current_question = question
        
        st.markdown("---")
        st.markdown("### 📈 About")
        st.markdown("""
        **CrediTrust Financial** serves 500,000+ customers across:
        - 🇰🇪 Kenya
        - 🇺🇬 Uganda  
        - 🇹🇿 Tanzania
        - 🇷🇼 Rwanda
        
        **Products:** Credit Cards, Personal Loans, BNPL, Savings, Money Transfers
        """)
        
        return {"product": selected_product if selected_product != "All Products" else None,
                "market": selected_market if selected_market != "All Markets" else None}

def render_main_content():
    """Render the main content area"""
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 Complaint Intelligence Dashboard")
    with col2:
        st.metric("Complaints Analyzed", "50,000+", "12% this month")
    
    st.markdown("""
    **Transform customer complaints into strategic insights.** Ask questions about product issues, 
    regional trends, operational challenges, and emerging patterns across our East African markets.
    """)
    
    # Question input
    question = st.text_input(
        "**💬 Ask a business question:**",
        placeholder="e.g., What are the top payment processing issues in Kenya?",
        value=st.session_state.get('current_question', ''),
        key="question_input"
    )
    
    # Filters from sidebar
    filters = {}
    if sidebar_filters['product']:
        filters['product'] = sidebar_filters['product']
    if sidebar_filters['market']:
        filters['market'] = sidebar_filters['market']
    
    # Display active filters
    if filters:
        filter_text = " | ".join([f"{k}: {v}" for k, v in filters.items()])
        st.info(f"**Active Filters:** {filter_text}")
    
    return question, filters

def render_results(answer, chunks, processing_time):
    """Render the analysis results"""
    # Answer section
    st.subheader("📈 Analysis Results")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Complaints Analyzed", len(chunks), "relevant excerpts")
    with col2:
        st.metric("Processing Time", f"{processing_time:.2f}s", "AI analysis")
    with col3:
        unique_products = set(chunk['metadata'].get('product', 'Unknown') for chunk in chunks)
        st.metric("Products Covered", len(unique_products))
    
    # Answer in expandable section
    with st.expander("🎯 Executive Summary", expanded=True):
        st.markdown(answer)
    
    # Source complaints
    if chunks:
        st.subheader("🔍 Source Complaints")
        
        for i, chunk in enumerate(chunks):
            with st.expander(f"Complaint #{i+1} | Product: {chunk['metadata'].get('product', 'Unknown')} | Market: {chunk['metadata'].get('market', 'Unknown')}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Excerpt:**\n{chunk['text']}")
                with col2:
                    st.metric("Relevance Score", f"{chunk.get('score', 0):.3f}")
                    st.caption(f"Product: {chunk['metadata'].get('product', 'N/A')}")
                    st.caption(f"Market: {chunk['metadata'].get('market', 'N/A')}")
                    st.caption(f"Channel: {chunk['metadata'].get('channel', 'N/A')}")

# ---------------------------
# Main App Logic
# ---------------------------
def main():
    # Initialize session state
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ''
    
    # Load pipeline (cached)
    rag_pipeline, global_config, validator = load_pipeline()
    
    if rag_pipeline is None:
        st.error("Failed to initialize the analysis system. Please check the logs.")
        return
    
    global config
    config = global_config
    
    # Render sidebar and get filters
    sidebar_filters = render_sidebar()
    
    # Render main content and get question
    question, filters = render_main_content()
    
    # Process question
    if question and st.button("🚀 Analyze Complaints", type="primary"):
        # Validate query
        is_valid, validation_msg = validator.validate_query(question)
        
        if not is_valid:
            st.warning(f"❌ {validation_msg}")
            with st.expander("💡 Suggested Business Questions"):
                st.markdown(validator.suggest_questions())
        else:
            # Process the question
            start_time = time.time()
            
            with st.spinner("🤖 Analyzing complaints across our markets..."):
                try:
                    answer, chunks = rag_pipeline.run(question, config.TOP_K_RETRIEVAL, filters)
                    processing_time = time.time() - start_time
                    
                    # Render results
                    render_results(answer, chunks, processing_time)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Streamlit app error: {e}")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption("Built with ❤️ for CrediTrust Financial | AI-Powered Complaint Intelligence")
    with col2:
        st.caption(f"v1.0 | {len(config.PRODUCTS)} products")
    with col3:
        st.caption(f"{len(config.MARKETS)} markets")

# ---------------------------
# Run the app
# ---------------------------
if __name__ == "__main__":
    main()