import pickle
from collections import Counter

def check_meta():
    try:
        with open('vector_store/credtrust_bi_index_meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        
        markets = Counter(m['market'] for m in meta)
        products = Counter(m['product'] for m in meta)
        
        print("\n--- Metadata Distribution ---")
        print(f"Total Chunks: {len(meta)}")
        print("\nMarkets:")
        for k, v in markets.items():
            print(f"  {k}: {v}")
            
        print("\nProducts:")
        for k, v in products.items():
            print(f"  {k}: {v}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_meta()
