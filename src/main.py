from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import kg_utils
import vectordb
import chunker
import inputs
import spacy 
import util
import time
import eval
import csv

# Load models
model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_lg")

# Orders query entities based on importance dynamically.
def prioritize_query_entities(entities, relations):
    priority_order = {
        "subject": 1,  
        "key_attribute": 2,  
        "related_actor": 3,  
        "source": 4  
    }

    # Count entity occurrences in relations to determine importance
    entity_counts = Counter()
    for rel in relations:
        entity_counts[rel['source']] += 1
        entity_counts[rel['target']] += 1

    # Classify entities dynamically
    entity_priority = {}
    for entity in entities:
        if entity in entity_counts and entity_counts[entity] > 1:
            entity_priority[entity] = priority_order["subject"]  # Frequently mentioned = Subject
        elif any(entity in rel['target'] for rel in relations):
            entity_priority[entity] = priority_order["key_attribute"]  # Appears in relations as target
        elif any(entity in rel['source'] for rel in relations):
            entity_priority[entity] = priority_order["related_actor"]  # Appears in relations as source
        else:
            entity_priority[entity] = priority_order["source"]  # Default to source

    # Sort entities by priority
    sorted_entities = sorted(entities, key=lambda e: (entity_priority[e], -entity_counts[e]))

    return sorted_entities

def rank_by_similarity_and_evidence(candidates, w_sim=0.5, w_evidence=0.3, w_depth=0.2, w_proximity_penalty=0.8):
    support = defaultdict(list)
    for c in candidates:
        support[c['target']].append(c)

    ranked_candidates = []

    for target, evidence_list in support.items():
        avg_similarity = sum(c['similarity'] for c in evidence_list) / len(evidence_list)
        min_depth = min(c['depth'] for c in evidence_list)
        evidence_count = len(evidence_list)

        norm_similarity = avg_similarity
        norm_depth = 1 / (min_depth + 1e-5)

        # Composite score per group
        composite_score = (
            w_sim * norm_similarity +
            w_evidence * evidence_count +
            w_depth * norm_depth
        )

        for c in evidence_list:
            # Penalize if source is the same as target (e.g., circular path)
            proximity_penalty = 1.0 if c["source"] == c["target"] else 0.0
            c['composite_score'] = composite_score - w_proximity_penalty * proximity_penalty
            ranked_candidates.append(c)

    ranked_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
    return ranked_candidates

# Generates an answer by traversing the knowledge graph.
def generate_answer_multihop(knowledge_graph, query, node_embeddings, edge_embeddings, max_depth=3, top_k=4):
    answer_candidates = []

    q_entities = kg_utils.extract_query_entities([query])
    query_entities = {util.clean(ent) for sublist in q_entities["entities"] for ent in sublist}
    query_relations = [rel for sublist in q_entities["relations"] for rel in sublist]

    query_entities = prioritize_query_entities(query_entities, query_relations)
    query_relation_texts = [f"{util.clean(r['source'])} {util.clean(r['relation'])} {util.clean(r['target'])}" for r in query_relations]

    targets = [util.clean(r['target']) for r in query_relations] + [util.clean(r['source']) for r in query_relations]

    print("QUERY formed by LLM: ", query_relation_texts)

    # Cache query relation embeddings
    query_relation_embeddings = {
        query_text: model.encode(query_text, convert_to_tensor=True)
        for query_text in query_relation_texts
    }

    # STOP_RELATIONS = {"of", "in", "during"}

    def get_similar_nodes(node_name, top_k=3):
        if not node_embeddings:
            return []

        query_emb = model.encode(node_name, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        node_ids, node_vecs = zip(*node_embeddings.items())
        node_vecs = np.array(node_vecs)

        sims = cosine_similarity(query_emb, node_vecs)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        return [(node_ids[i], sims[i]) for i in top_indices]


    target_like_nodes = []
    for target in targets:
        target_like_nodes.append(target)
        matching_nodes = get_similar_nodes(target, top_k=3)
        # print(f"Target: {target}, Matching nodes: {matching_nodes}")
        target_like_nodes.extend([n for n, _ in matching_nodes])
    target_like_nodes = list(set(target_like_nodes))  # Deduplicate

    # print("Matching KG nodes for query targets:", target_like_nodes)

    for matched_target in target_like_nodes:
        visited = set()
        queue = [([matched_target], [])]  # (path_nodes, path_edges)

        while queue:
            current_path, edge_path = queue.pop(0)
            current_node = current_path[-1]

            if len(edge_path) >= max_depth:
                continue

            for _, neighbor, data in knowledge_graph.edges(current_node, data=True):  # reversed: tracing back to source
                relation = data.get("relation", "related_to").lower()
                if neighbor in visited:
                    continue

                new_edge = (neighbor, relation, current_node)
                new_path = [neighbor] + current_path
                new_edge_path = [new_edge] + edge_path

                # Semantic path representation
                path_text = "; ".join([f"{r} {v}" for (_, r, v) in new_edge_path])
                path_embedding = model.encode(path_text, convert_to_tensor=True)

                # Match against query relations
                max_similarity = -1
                for query_text, query_embedding in query_relation_embeddings.items():
                    sim_score = cosine_similarity(
                        path_embedding.cpu().numpy().reshape(1, -1),
                        query_embedding.cpu().numpy().reshape(1, -1)
                    )[0][0]
                    if sim_score > max_similarity:
                        max_similarity = sim_score

                if max_similarity > 0.65:
                    print("Matched Path:", path_text, "→ Similarity:", max_similarity, "→ Source:", new_path[0], f"→ Depth: {len(new_edge_path)}\n")

                if max_similarity > 0.60:
                    answer_candidates.append({
                        "source": new_path[0],
                        "target": matched_target,
                        "similarity": max_similarity,
                        "path": new_edge_path,
                        "depth": len(new_edge_path),
                    })

                queue.append((new_path, new_edge_path))
                visited.add(neighbor)

    ranked_candidates = rank_by_similarity_and_evidence(answer_candidates)
    print("Knowledge Graph Candidates:", ranked_candidates[:top_k])

    if ranked_candidates:
        return ranked_candidates[0]["source"]
    return ""


if __name__ == "__main__": 
    df, test_data = inputs.load_data()
    category_test_data, category_urls = inputs.prepare_data_by_category(test_data)
    category_corpus = inputs.filter_corpus_by_categories_subset(df, category_urls)
    chunker.chunk_all_categories()
    chunks_data = chunker.load_chunks()
    vectordb.index_all_categories(chunks_data)
    graphs_by_category = kg_utils.extract_entities_and_create_graph(chunks_data)
    kg_utils.plot_and_save_category_knowledge_graph(graphs_by_category)

    graphs_by_category = kg_utils.load_all_knowledge_graphs()
    node_embeddings_per_category, edge_embeddings_per_category = vectordb.vectorize_knowledge_graph_per_category(graphs_by_category)  
    vectordb.plot_node_embeddings(node_embeddings_per_category)  
    faiss_node_indices, faiss_edge_indices, edge_id_maps = vectordb.store_embeddings_in_faiss_per_category(node_embeddings_per_category, edge_embeddings_per_category)
    vectordb.save_embeddings_and_faiss_index(node_embeddings_per_category, edge_embeddings_per_category, faiss_node_indices, faiss_edge_indices, edge_id_maps)
    
    # Inference
    faiss_indices = vectordb.load_all_faiss_indices("faiss_index")
    node_embeddings, edge_embeddings, faiss_node_indices, faiss_edge_indices, edge_id_maps = vectordb.load_embeddings_and_faiss_index()

    qna_data = inputs.load_qna_from_category_files()
    eval_results = []
    total_time = 0
    query_count = 0

    for category, qna_list in qna_data.items():
        # Load finalized questions once
        print(f"\nCategory: {category}")
        results = []
        filename = f"data/subset/results2/{category}.csv"
        for q, a in qna_list:
            t = "KG"
            print(f"Q: {q}")
            print(f"Expected A: {a}\n")

            G = graphs_by_category[category]
            start_time = time.time()
            print("Step 1: Knowledge Graph retrieval!!\n")
            answer = generate_answer_multihop(G, q, node_embeddings[category], edge_embeddings[category])
            
            if not answer:
                print("\nStep 2: KG retrieval failed. Trying dense retrieval!!!!\n")
                semantic_retriever = faiss_indices[category]
                chunks = vectordb.retrieve_from_faiss(q, semantic_retriever)
                context = " ".join(chunks)
                print("Context chunk retrieved from Vector database:\n", context)
                answer = util.llm_answer_with_context(q, context)
                t = "LLM"

            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            print(f"\nGenerated A: {answer}, Inference Type: {t}\n")
            result = {
                "question": q, 
                "expected_answer": a,
                "generated_answer": answer,
                "inference_type": t
            }
            eval_results.append({
                "question": result["question"],
                "gold_answer": result["expected_answer"],
                "predicted_answer": result["generated_answer"],
                "exact_match": eval.exact_match(result["generated_answer"], result["expected_answer"]),
                "kg_hit": eval.kg_hit_rate(result["inference_type"]),
                # "context_precision": eval.context_precision(context, a),
                # "context_recall": eval.context_recall(context, a),
                # "top_5_accuracy": eval.top_k_accuracy(context, a, k=5),
                # "inference_time": elapsed_time
            })

            results.append(result)

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["question", "expected_answer", "generated_answer", "inference_type"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        df = pd.DataFrame(eval_results)
        print(len(eval_results))
        print("\n--- Evaluation Summary ---")
        print(f"Exact Match: {df['exact_match'].mean():.2f}")
        print(f"KG Hit Rate: {df['kg_hit'].mean():.2f}")
        # print(f"Context Precision (avg): {df['context_precision'].mean():.2f}")
        # print(f"Context Recall: {df['context_recall'].mean():.2f}")
        # print(f"Top-5 Retrieval Accuracy: {df['top_5_accuracy'].mean():.2f}")
        # print(f"Avg Inference Time: {df['inference_time'].mean():.2f} sec")
        break
