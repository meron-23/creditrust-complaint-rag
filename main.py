import os
import argparse
from src.config import Config
from src.preprocessing import load_complaints, preprocess_dataset
from src.indexer import ComplaintIndexer
from src.retriever import ComplaintRetriever
from src.generator import BusinessAnswerGenerator
from src.rag_pipeline import RAGPipeline
from src.utils.logger import setup_logger
from src.query_validator import QueryValidator

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="CrediTrust Financial - Complaint Insights AI")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild vector index")
    parser.add_argument("--question", type=str, help="Business question to analyze")
    parser.add_argument("--product", type=str, help="Filter by product type")
    parser.add_argument("--market", type=str, help="Filter by market/country")
    args = parser.parse_args()
    
    config = Config.from_env()
    validator = QueryValidator(config)
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing CrediTrust complaint data...")
        df = load_complaints(config.DATA_PATH)
        df = preprocess_dataset(df)
        
        # Initialize indexer
        indexer = ComplaintIndexer(config)
        
        # Build or load index
        if args.rebuild_index or not os.path.exists(config.VECTOR_STORE_PATH + ".index"):
            logger.info("Building new business intelligence index...")
            indexer.build_index(df)
            indexer.save()
        else:
            logger.info("Loading existing business index...")
            indexer.load()
        
        # Initialize retriever and generator
        retriever = ComplaintRetriever(indexer.model, indexer.index, indexer.metadatas)
        generator = BusinessAnswerGenerator(config)  # Pass config to generator
        rag = RAGPipeline(retriever, generator)
        
        # Apply business filters if provided
        filters = {}
        if args.product:
            filters['product'] = args.product
            logger.info(f"Applying product filter: {args.product}")
        if args.market:
            filters['market'] = args.market
            logger.info(f"Applying market filter: {args.market}")
        
        # Interactive mode or single question
        if args.question:
            answer, context_chunks = rag.run(args.question, config.TOP_K_RETRIEVAL, filters)
            print(f"\n🔍 **Business Question**: {args.question}")
            print("\n📈 **ANALYSIS RESULTS**")
            print("=" * 60)
            print(answer)
            print("=" * 60)
            
            if context_chunks and "I couldn't find" not in answer and "error" not in answer.lower():
                print(f"\n📋 **Source Data**: {len(context_chunks)} complaint excerpts analyzed")
                products = set(chunk['metadata'].get('product', 'Unknown') for chunk in context_chunks)
                markets = set(chunk['metadata'].get('market', 'Unknown') for chunk in context_chunks)
                print(f"   Products: {', '.join(products)}")
                print(f"   Markets: {', '.join(markets)}")
        
        else:
            # Interactive business intelligence mode
            print("\n🤖 **CrediTrust Financial - Complaint Insights AI**")
            print("📍 Serving Kenya, Uganda, Tanzania, Rwanda")
            print("📊 Analyzing: Credit Cards, Personal Loans, BNPL, Savings, Money Transfers")
            print("\n💡 Type 'exit' to quit, 'help' for guidance, 'examples' for business questions")
            
            while True:
                try:
                    question = input("\n🔍 **Business Question**: ").strip()
                    
                    if question.lower() == 'exit':
                        print("\n👋 Thank you for using CrediTrust Insights AI. Goodbye!")
                        break
                    elif question.lower() == 'help':
                        print("\n💼 **I help CrediTrust teams analyze customer complaints for:**")
                        print("  • Product issue identification and trending")
                        print("  • Regional market analysis across East Africa")
                        print("  • Operational and process improvement opportunities")
                        print("  • Regulatory and compliance insights")
                        print("  • Customer experience enhancement")
                        print("  • Competitive intelligence from user feedback")
                        continue
                    elif question.lower() == 'examples':
                        print(validator.suggest_questions())
                        continue
                    elif not question:
                        continue
                    
                    # Validate business query first
                    is_valid, validation_msg = validator.validate_query(question)
                    if not is_valid:
                        print(f"\n❌ {validation_msg}")
                        continue
                    
                    # Run business analysis
                    answer, context_chunks = rag.run(question, config.TOP_K_RETRIEVAL, filters)
                    
                    print(f"\n📈 **ANALYSIS RESULTS**")
                    print("=" * 60)
                    print(answer)
                    print("=" * 60)
                    
                    if context_chunks and "I couldn't find" not in answer and "error" not in answer.lower():
                        print(f"\n📋 **Source Data**: {len(context_chunks)} complaint excerpts analyzed")
                        products = set(chunk['metadata'].get('product', 'Unknown') for chunk in context_chunks)
                        markets = set(chunk['metadata'].get('market', 'Unknown') for chunk in context_chunks)
                        print(f"   Products: {', '.join(products)}")
                        print(f"   Markets: {', '.join(markets)}")
                        
                        # Show sample of most relevant complaint
                        if context_chunks:
                            print(f"\n🔎 **Sample Complaint Excerpt**:")
                            print(f"Product: {context_chunks[0]['metadata'].get('product', 'Unknown')}")
                            print(f"Market: {context_chunks[0]['metadata'].get('market', 'Unknown')}")
                            print(f"Text: {context_chunks[0]['text'][:100]}...")
                
                except KeyboardInterrupt:
                    print("\n👋 Thank you for using CrediTrust Insights AI. Goodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in business analysis mode: {e}")
                    print("❌ Sorry, I encountered an error processing your business question. Please try again.")
    
    except Exception as e:
        logger.error(f"CrediTrust application failed: {e}")
        print(f"❌ Application error: {e}")

def show_available_filters(metadatas):
    """Show available business filter options"""
    products = set()
    markets = set()
    channels = set()
    
    for meta in metadatas:
        if 'product' in meta:
            products.add(meta['product'])
        if 'market' in meta:
            markets.add(meta['market'])
        if 'channel' in meta:
            channels.add(meta['channel'])
    
    print("\n🎯 **Available Business Filters:**")
    print(f"Products: {', '.join(sorted(products))}")
    print(f"Markets: {', '.join(sorted(markets))}")
    print(f"Channels: {', '.join(sorted(channels))}")
    print("\n💡 Use --product <product_name> or --market <country_name> in command line")

if __name__ == "__main__":
    main()