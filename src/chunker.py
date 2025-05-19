import os
import json
import regex as re
import pandas as pd
from transformers import pipeline, AutoTokenizer
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from sentence_transformers import SentenceTransformer, util


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95)

model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name, device=-1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    lines = text.split('. ')
    cleaned_lines = [
        line for line in lines 
        if not re.match(r'^[A-Z\s]{5,}$', line.strip()) and 
           not line.strip().endswith('?') and 
           not line.strip().endswith(':')
    ]
    return '. '.join(cleaned_lines).strip()

def chunk_text_by_tokens(text, tokenizer, max_tokens=1024):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=False)[0]
    chunks = []
    
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks

def summarize_document(document_text, tokenizer, summarizer, max_input_tokens=1000, max_output_length=200):
    chunks = chunk_text_by_tokens(document_text, tokenizer, max_tokens=max_input_tokens)
        
    summaries = []
    
    for i, chunk in enumerate(chunks):
        try:
            tokenized = tokenizer(chunk, return_tensors="pt", truncation=False)
            summary = summarizer(chunk, max_length=min(max_output_length, len(tokenized['input_ids'][0])), min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Skipping chunk {i} due to error: {e}")

    return ' '.join(summaries)

def semantic_chunk_text(text, max_tokens=100, similarity_threshold=0.75):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        chunk_text = '. '.join(current_chunk)
        token_count = len(chunk_text.split())

        if token_count > max_tokens:
            chunks.append('. '.join(current_chunk[:-1]))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append('. '.join(current_chunk))

    # Merge small chunks based on similarity
    merged_chunks = []
    i = 0
    while i < len(chunks):
        if i < len(chunks) - 1:
            emb1 = embedding_model.encode(chunks[i], convert_to_tensor=True)
            emb2 = embedding_model.encode(chunks[i + 1], convert_to_tensor=True)
            sim = util.pytorch_cos_sim(emb1, emb2).item()

            if sim > similarity_threshold:
                merged_chunks.append(chunks[i] + ". " + chunks[i + 1])
                i += 2
                continue

        merged_chunks.append(chunks[i])
        i += 1

    return merged_chunks

def semantic_chunk_category_file(csv_path, category_name=None, output_path=None):
    df = pd.read_csv(csv_path)

    summarized_texts = []
    for _, row in df.iterrows():
        if category_name and row.get("category", "").lower() != category_name.lower():
            continue

        title = str(row.get("title", ""))
        body = str(row.get("body", ""))
        body = clean_text(body)
        content = f"{title}\n\n{body}".strip()

        summarized_content = summarize_document(content, tokenizer, summarizer)
        summarized_texts.append(summarized_content)

    combined_summary = "\n\n".join(summarized_texts)

    documents = []
    try:
        semantic_chunks = semantic_chunker.create_documents([combined_summary])
        for chunk in semantic_chunks:
            documents.append(Document(page_content=chunk.page_content))
    except Exception as e:
        print(f"Error during semantic chunking: {e}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        document_data = [
            {
                "chunk": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(document_data, f, ensure_ascii=False, indent=4)

    return documents

def chunk_all_categories(input_dir="data/subset/corpus", output_dir="data/subset/cleaned_chunks"):
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for csv_file in csv_files:
        category_name = csv_file.replace(".csv", "")
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, f"{category_name}.txt")

        print(f"Processing category: {category_name}")
        semantic_chunks = semantic_chunk_category_file(
            csv_path=input_path,
            category_name=category_name,
            output_path=output_path
        )

def load_chunks(input_dir="data/subset/cleaned_chunks"):
    category_chunks = {}

    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return category_chunks

    chunk_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    # Process each chunk file
    for chunk_file in chunk_files:
        category_name = chunk_file.replace(".txt", "") 
        file_path = os.path.join(input_dir, chunk_file)

        print(f"Loading chunks from category: {category_name}")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                chunks = json.load(file)
                
                # Collect the chunks for this category
                category_chunks[category_name] = []
                for chunk_data in chunks:
                    chunk_content = chunk_data.get("chunk", "").strip()
                    metadata = chunk_data.get("metadata", {})

                    if chunk_content:
                        category_chunks[category_name].append({
                            "chunk": chunk_content,
                            "metadata": metadata
                        })
                if not category_chunks[category_name]:
                    print(f"No valid chunks found in {chunk_file}")
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {chunk_file}: {e}")
        except Exception as e:
            print(f"Error loading {chunk_file}: {e}")

    print(f"Loaded {sum(len(chunks) for chunks in category_chunks.values())} chunks across categories.")
    return category_chunks