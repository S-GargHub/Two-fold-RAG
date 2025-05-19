import os
import faiss
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from sentence_transformers import SentenceTransformer


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedding = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def index_chunks_to_faiss(category_name, chunks, save_path="faiss_index"):
    # Now index using LangChain FAISS
    docs = [Document(page_content=chunk["chunk"]) for chunk in chunks]

    if not docs:
        print(f"No valid documents found for category: {category_name}. Skipping indexing.")
        return None

    vectorstore = FAISS.from_documents(docs, embedding)

    category_path = os.path.join(save_path, category_name)
    os.makedirs(category_path, exist_ok=True)
    vectorstore.save_local(category_path)
    print(f"Saved FAISS index for {category_name} to {category_path}")
    return vectorstore

def index_all_categories(category_chunks, save_path="faiss_index"):
    # Find all chunk files in the input directory
     for category_name, chunks in category_chunks.items():
        print(f"Indexing chunks for category: {category_name}")
        index_chunks_to_faiss(category_name, chunks, save_path)

def load_faiss_index_for_category(category_name, base_path="faiss_index"):
    category_path = os.path.join(base_path, category_name)
    if not os.path.exists(category_path):
        raise FileNotFoundError(f"FAISS index path not found for category: {category_name}")

    retriever = FAISS.load_local(category_path, embedding, allow_dangerous_deserialization=True).as_retriever()
    return retriever

def load_all_faiss_indices(save_path="faiss_index"):
    faiss_indices = {}
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Save path {save_path} does not exist.")

    for category_name in os.listdir(save_path):
        category_path = os.path.join(save_path, category_name)
        if os.path.isdir(category_path):
            try:
                faiss_indices[category_name] = load_faiss_index_for_category(category_name, save_path)
                print(f"Loaded FAISS index for category: {category_name}")
            except Exception as e:
                print(f"Failed to load FAISS index for {category_name}: {e}")
    
    return faiss_indices

def retrieve_from_faiss(query, vector_db, index_path="faiss_index", top_k=5):
    results = vector_db.invoke(query, config={"k": top_k})
    retrieved_chunks = [doc.page_content for doc in results]
    return retrieved_chunks

def vectorize_knowledge_graph(knowledge_graph, model_name="all-MiniLM-L6-v2"):
    """
    Vectorizes the knowledge graph using SentenceTransformer embeddings.
    Nodes are encoded based on their text. Edges are averaged embeddings of their nodes.
    """
    model = SentenceTransformer(model_name)

    node_ids = list(knowledge_graph.nodes)
    
    # Encode all node IDs (or labels)
    node_vectors = model.encode(node_ids, convert_to_numpy=True, batch_size=64)
    node_embeddings = dict(zip(node_ids, node_vectors))

    # Compute edge embeddings by averaging node embeddings
    edge_embeddings = {}
    for u, v in knowledge_graph.edges():
        if u in node_embeddings and v in node_embeddings:
            edge_embeddings[(u, v)] = np.mean([node_embeddings[u], node_embeddings[v]], axis=0)

    return node_embeddings, edge_embeddings

def store_embeddings_in_faiss_per_category(node_embeddings_per_category, edge_embeddings_per_category, dimension=384):
    faiss_node_indices = {}
    faiss_edge_indices = {}
    edge_id_maps = {}

    categories = set(node_embeddings_per_category.keys()).union(edge_embeddings_per_category.keys())

    for category in categories:
        # --- Node Embeddings ---
        if category in node_embeddings_per_category:
            node_embeddings = node_embeddings_per_category[category]
            node_matrix = np.array(list(node_embeddings.values()), dtype=np.float32)
            node_index = faiss.IndexFlatL2(dimension)
            node_index.add(node_matrix)
            faiss_node_indices[category] = node_index

        # --- Edge Embeddings ---
        if category in edge_embeddings_per_category:
            edge_embeddings = edge_embeddings_per_category[category]
            edge_matrix = np.array(list(edge_embeddings.values()), dtype=np.float32)
            edge_index = faiss.IndexFlatL2(dimension)
            edge_index.add(edge_matrix)
            faiss_edge_indices[category] = edge_index

            # Edge ID map: FAISS returns index positions only
            edge_id_maps[category] = {i: edge for i, edge in enumerate(edge_embeddings.keys())}

    return faiss_node_indices, faiss_edge_indices, edge_id_maps

def vectorize_knowledge_graph_per_category(graphs_by_category):
    node_embeddings_per_category = {}
    edge_embeddings_per_category = {}

    for category, graph in graphs_by_category.items():
        if len(graph.nodes()) == 0:
            print(f"Skipping category '{category}' — empty graph.")
            continue
        print("Processing category:", category)
        assert all(isinstance(n, str) for n in graph.nodes())
        node_embeddings, edge_embeddings = vectorize_knowledge_graph(graph) 
        node_embeddings_per_category[category] = node_embeddings
        edge_embeddings_per_category[category] = edge_embeddings
    
    return node_embeddings_per_category, edge_embeddings_per_category

def plot_node_embeddings(node_embeddings_per_category, output_dir="embedding_plots", perplexity=30, n_iter=500):
    os.makedirs(output_dir, exist_ok=True)

    for category, node_embeddings in node_embeddings_per_category.items():
        if len(node_embeddings) == 0:
            print(f"Skipping plot for '{category}' — no embeddings.")
            continue

        print(f"Plotting and saving node embeddings for category: {category}")

        labels = list(node_embeddings.keys())
        vectors = np.array(list(node_embeddings.values()))

        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], s=100, alpha=0.7)

        for i, label in enumerate(labels):
            plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], label, fontsize=9, ha='right', va='bottom')

        plt.title(f"t-SNE of Node Embeddings for Category: {category}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{category}_node_embeddings.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved plot to: {output_path}")

def save_embeddings_and_faiss_index(
    node_embeddings_per_category,
    edge_embeddings_per_category,
    faiss_node_indices,
    faiss_edge_indices,
    edge_id_maps,
    save_path="saved_embeddings"
):
    os.makedirs(save_path, exist_ok=True)

    # Save node and edge embeddings
    with open(os.path.join(save_path, "node_embeddings.pkl"), "wb") as f:
        pickle.dump(node_embeddings_per_category, f)

    with open(os.path.join(save_path, "edge_embeddings.pkl"), "wb") as f:
        pickle.dump(edge_embeddings_per_category, f)

    # Save FAISS node indices
    for category, index in faiss_node_indices.items():
        faiss.write_index(index, os.path.join(save_path, f"faiss_node_index_{category}.faiss"))

    # Save FAISS edge indices
    for category, index in faiss_edge_indices.items():
        faiss.write_index(index, os.path.join(save_path, f"faiss_edge_index_{category}.faiss"))

    # Save edge ID maps
    with open(os.path.join(save_path, "edge_id_maps.pkl"), "wb") as f:
        pickle.dump(edge_id_maps, f)

    print(f"✅ Embeddings and FAISS indices saved to '{save_path}'")

def load_embeddings_and_faiss_index(save_path="saved_embeddings"):
    # Load node and edge embeddings
    with open(os.path.join(save_path, "node_embeddings.pkl"), "rb") as f:
        node_embeddings_per_category = pickle.load(f)

    with open(os.path.join(save_path, "edge_embeddings.pkl"), "rb") as f:
        edge_embeddings_per_category = pickle.load(f)

    with open(os.path.join(save_path, "edge_id_maps.pkl"), "rb") as f:
        edge_id_maps = pickle.load(f)

    # Load FAISS indices
    faiss_node_indices = {}
    faiss_edge_indices = {}

    for fname in os.listdir(save_path):
        if fname.startswith("faiss_node_index_") and fname.endswith(".faiss"):
            category = fname.replace("faiss_node_index_", "").replace(".faiss", "")
            faiss_node_indices[category] = faiss.read_index(os.path.join(save_path, fname))

        elif fname.startswith("faiss_edge_index_") and fname.endswith(".faiss"):
            category = fname.replace("faiss_edge_index_", "").replace(".faiss", "")
            faiss_edge_indices[category] = faiss.read_index(os.path.join(save_path, fname))

    print(f"✅ Loaded embeddings and FAISS indices from '{save_path}'")
    return (
        node_embeddings_per_category,
        edge_embeddings_per_category,
        faiss_node_indices,
        faiss_edge_indices,
        edge_id_maps
    )