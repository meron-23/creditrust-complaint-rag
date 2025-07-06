# CrediTrust Complaint Intelligence (RAG System)

This project is an internal AI tool for CrediTrust Financial, designed to help teams understand and act on customer complaints quickly and efficiently using Retrieval-Augmented Generation (RAG). It transforms thousands of unstructured complaint narratives into clear, actionable insights using semantic search and language models.

## Business Objective

CrediTrust operates across East Africa with over 500,000 users. Internal teams currently face delays and bottlenecks due to the manual handling of complaint data. This project aims to:

- Reduce the time it takes to identify complaint trends from days to minutes.
- Empower non-technical teams (Support, Compliance, Product) to ask plain-English questions and get evidence-backed answers.
- Shift the company from reacting to issues to proactively identifying and fixing them.

## Solution Overview

We build a **RAG (Retrieval-Augmented Generation)** pipeline that:

- Uses semantic embeddings + vector search (FAISS or ChromaDB) to retrieve relevant complaints.
- Feeds those into a large language model (LLM) to generate concise summaries and answers.
- Supports filtering by product category (Credit Card, Personal Loan, BNPL, Savings Account, Money Transfer).

---

## Task 1: EDA & Preprocessing

We begin by analyzing and cleaning the CFPB consumer complaint dataset to extract valuable insights and prepare it for embedding.

### Completed in this stage:
- Loaded full dataset from CFPB.
- Analyzed complaint counts across products.
- Visualized narrative lengths and distributions.
- Filtered down to 5 product categories relevant to CrediTrust.
- Removed entries with missing narratives.
- Cleaned text (lowercasing, punctuation removal, boilerplate stripping).
- Saved final cleaned dataset to `data/filtered_complaints.csv`.

---