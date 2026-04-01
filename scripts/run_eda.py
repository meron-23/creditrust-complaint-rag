import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import map_product_to_group

def run_eda(data_path="data/filtered_complaints.csv"):
    print(f"--- CrediTrust Business Intelligence: EDA Report ---")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    # Load data
    df = pd.read_csv(data_path, low_memory=False)
    total_rows = len(df)
    
    # 1. Product Distribution
    df['product_group'] = df['Product'].apply(map_product_to_group)
    product_counts = df['product_group'].value_counts()
    product_pct = df['product_group'].value_counts(normalize=True) * 100
    
    # 2. Narrative Length Analysis
    # Ensure Consumer complaint narrative is string
    df['narrative_len'] = df['Consumer complaint narrative'].astype(str).apply(lambda x: len(x.split()))
    avg_len = df['narrative_len'].mean()
    median_len = df['narrative_len'].median()
    max_len = df['narrative_len'].max()
    min_len = df['narrative_len'].min()
    
    # 3. Missing Narratives
    missing_narratives = df['Consumer complaint narrative'].isna().sum()
    
    # 4. Visualization (Optional - saves to file if possible, or just print stats)
    # We'll stick to text output for this script so the user can see it in terminal
    
    print(f"\n1. DATASET OVERVIEW")
    print(f"   Total complaints loaded: {total_rows:,}")
    print(f"   Complaints with missing narratives: {missing_narratives:,} ({missing_narratives/total_rows*100:.1f}%)")
    
    print(f"\n2. PRODUCT DISTRIBUTION")
    for prod, count in product_counts.items():
        print(f"   - {prod}: {count:,} ({product_pct[prod]:.1f}%)")
        
    print(f"\n3. COMPLAINT DEPTH (Word Count)")
    print(f"   Average narrative length: {avg_len:.1f} words")
    print(f"   Median narrative length: {median_len:.1f} words")
    print(f"   Range: {min_len} to {max_len} words")
    
    print(f"\n4. EXECUTIVE SUMMARY FINDINGS")
    print("   The dataset represents a robust cross-section of CrediTrust Financial's product ecosystem. ")
    print(f"   {product_counts.index[0]} emerges as the primary driver of customer feedback, accounting for {product_pct.iloc[0]:.1f}% of the volume.")
    print(f"   The high average narrative length of {avg_len:.1f} words suggests that customers are providing detailed, ")
    print("   unstructured feedback that is ripe for semantic analysis rather than simple keyword matching.")
    print("   This confirms the necessity of a RAG-based approach to extract nuanced business insights from raw narratives.")

if __name__ == "__main__":
    run_eda()
