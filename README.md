## Two-fold-RAG
A novel approach to Retrieval-Augmented Generation (RAG) that combines knowledge graph traversal and traditional dense vector retrieval to enhance LLM responses.

## Overview
Two-fold-RAG improves upon traditional RAG systems by implementing a dual retrieval approach:

1. Knowledge Graph Traversal: Primary attempt to retrieve answers by traversing relationship paths through a semantically-rich knowledge graph
2. Vector Similarity Fallback: Secondary retrieval using traditional vector similarity search when knowledge graph traversal fails to yield results

This hybrid approach significantly improves the quality and relevance of information provided to Large Language Models (LLMs), resulting in more accurate, context-aware responses.

## Features

🔍 Dual Retrieval Strategy: Knowledge graph traversal with vector similarity fallback 

🧠 Semantic Knowledge Graphs: Automatically extracts entities and relationships from corpus documents

🔄 Multi-hop Reasoning: Supports complex multi-hop queries by traversing relationship paths

📊 Entity Prioritization: Dynamically prioritizes query entities based on semantic importance

💯 Relevance Scoring: Implements custom scoring algorithms for candidate answers

🔎 Vector Embeddings: Utilizes sentence transformers for high-quality semantic embeddings

📈 Visualization: Includes tools for knowledge graph visualization and embedding space analysis

## Architecture
Two-fold-RAG implements the following process flow:

1. Data Preparation: Process corpus documents by category and summarize them
2. Chunking: Create semantic chunks of appropriate length
3. Knowledge Graph Creation: Extract entities and relationships to build knowledge graphs
4. Vectorization: Generate embeddings for both documents and knowledge graph elements

### Query Processing:

1. Extract entities and relationships from user query
2. Attempt knowledge graph traversal to find paths that answer the query
3. Fall back to traditional vector retrieval if needed
4. Generate a final answer using an LLM


## Dataset
The system uses the MultiHopRAG dataset, which can be loaded using the Hugging Face datasets library:

```
from datasets import load_dataset
ds = load_dataset("yixuantt/MultiHopRAG", "corpus")
ds_q = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
```

## Installation
```
# Clone the repository
git clone https://github.com/S-GargHub/Two-fold-RAG.git
cd Two-fold-RAG

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
