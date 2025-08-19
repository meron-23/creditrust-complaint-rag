# 🤖 CrediTrust Financial - Complaint Intelligence AI

> Transforming customer complaints into strategic business insights across East Africa

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange?style=for-the-badge)](https://faiss.ai/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/)

## 🌟 Overview

CrediTrust Financial is a fast-growing digital finance company serving **500,000+ customers** across **Kenya, Uganda, Tanzania, and Rwanda**. This AI-powered complaint intelligence system transforms unstructured customer feedback into actionable business insights, enabling product teams to quickly identify issues, track trends, and make data-driven decisions.

**Key Products Analyzed:**
- 💳 Credit Cards
- 🏦 Personal Loans  
- 📱 BNPL (Buy Now, Pay Later)
- 💰 Savings Accounts
- 🔄 Money Transfers

## 🚀 Features

### 🔍 Intelligent Complaint Analysis
- **Natural Language Querying**: Ask business questions in plain English
- **Semantic Search**: Find relevant complaints using AI embeddings
- **Multi-market Analysis**: Compare trends across East African countries
- **Product-specific Insights**: Drill down into specific financial products

### 📊 Business Intelligence
- **Executive Summaries**: AI-generated insights with quantitative analysis
- **Trend Identification**: Spot emerging issues before they escalate
- **Regional Comparisons**: Understand geographic variations in complaints
- **Actionable Recommendations**: Data-driven suggestions for improvement

### 🎯 Built for Product Teams
- **PM-Friendly Interface**: Designed for Asha (BNPL Product Manager) and team
- **Real-time Filtering**: Focus on specific products or markets
- **Example Questions**: One-click access to common business queries
- **Transparent Sources**: See exactly which complaints informed each analysis

## 🏗️ Architecture

```mermaid
graph TB
    A[Customer Complaints] --> B[Data Preprocessing]
    B --> C[FAISS Vector Store]
    C --> D[Semantic Search]
    E[Business Question] --> D
    D --> F[Relevant Complaints]
    F --> G[LLM Analysis]
    G --> H[Strategic Insights]
```
## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Convert text to numerical vectors for semantic search |
| **Vector Database** | FAISS (Facebook AI Similarity Search) | Efficient similarity search and retrieval |
| **LLM** | `google/flan-t5-base` | Answer generation and analysis |
| **Frontend** | Streamlit | Interactive web dashboard |
| **Text Processing** | LangChain | Text splitting and chunking |
| **Machine Learning** | PyTorch | Model inference and processing |
| **Data Processing** | pandas | Data manipulation and cleaning |
| **Numerical Computing** | NumPy | Mathematical operations |

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Package Manager**: pip
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free disk space
- **Internet Connection**: For model downloads

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/credtrust/complaint-intelligence-ai.git
cd complaint-intelligence-ai
```
#### 1. Clone the Repository
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\
```
