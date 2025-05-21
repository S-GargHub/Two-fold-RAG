import os
import util
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

   
def extract_query_entities(texts):
    data = {'text': [], 'entities': [], 'relations': []}

    for text in texts:

        prompt = f"""
        You are an expert knowledge graph extraction agent.

        **Objective:**  
        Given the following question, extract only the relevant entities and relationships that are required to answer it. Avoid listing all possible entities and relations — only include those critical to finding the answer.

        **Definitions:**
        - **Entity**: A person, organization, event, or key concept that plays a role in the question.
        - **Relationship**: A verb or interaction that directly connects the entities necessary for reasoning toward the answer.

        **Output format (JSON only):**
        {{
            "entities": [{{"name": "entity1"}}, {{"name": "entity2"}}],
            "relationships": [{{"source": "entity1", "target": "entity2", "relation": "relationshipType"}}]
        }}

        **Example:**

        Question: "Who did TechCrunch report as being charged with fraud?"
        Output:
        {{
            "entities": [{{"name": "individual"}}, {{"name": "fraud"}}],
            "relationships": [
                {{"source": "individual", "target": "fraud", "relation": "charged with"}}
            ]
        }}

        **Now extract for:**
        "{text}"
        """
        
        response = util.call_llm(prompt)
        try:
            result = json.loads(response.text)
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            entity_names = [ent["name"] for ent in entities]
            data['text'].append(text)
            data['entities'].append(entity_names)
            data['relations'].append(relationships)
        except:
            print("Error processing text")

    sp_df = pd.DataFrame(data)
    return sp_df


def extract_entities_with_llm(texts):
    data = {'text': [], 'entities': [], 'relations': []}

    for text in texts:

        prompt = f"""
        **Task:** 
        Analyze the following document and identify all the named entities and relationships.
        
        **Entities (nodes):**
        - Identify key entities such as people, organizations, locations, events, concepts, and objects. 
        - Do not rely on any predefined list; dynamically identify the entities from the text.

        **Relationships (edges):**
        - Identify the relationships between entities based on verbs and interactions described in the text. 
        - Relationships should be meaningful and contextually relevant (e.g., 'works for', 'located in', 'owns', 'collaborates with', 'testified against').

        **Instructions:**
        1. Extract all entities in lowercase.
        2. Identify the relationships and link the entities with those relationships.
        3. Return the output in the following JSON format:

        {{
            "entities": [{{"name": "entity1"}}, {{"name": "entity2"}}],
            "relationships": [{{"source": "entity1", "target": "entity2", "relation": "relationshipType"}}]
        }}

        **Text:**
        "{text}"
        """
        
        response = util.call_llm(prompt)
        try:
            result = json.loads(response.text)
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            entity_names = [ent["name"] for ent in entities]
            data['text'].append(text)
            data['entities'].append(entity_names)
            data['relations'].append(relationships)
        except:
            print("Error processing text")

    sp_df = pd.DataFrame(data)
    return sp_df


#  Generates a knowledge graph from a list of text chunks 
def create_knowledge_graph(chunks, output_file=None):
    print("Generating knowledge graph!!")
    knowledge_graph = nx.Graph()
    
    df = extract_entities_with_llm(chunks)
    
    # Add relationships to the knowledge graph
    for rel_list in df["relations"]:
        for rel in rel_list:
            source = rel.get("source")
            target = rel.get("target")
            relation = rel.get("relation", "related_to")

            # Avoid self-loops and invalid relationships
            if source and target and source != target:
                knowledge_graph.add_edge(util.clean(source), util.clean(target), relation=util.clean(relation))
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(nx.node_link_data(knowledge_graph), f, ensure_ascii=False, indent=4)

    return knowledge_graph

def extract_entities_and_create_graph(chunks_data, output_dir="temp_KG"):
    os.makedirs(output_dir, exist_ok=True)
    knowledge_graphs = {}

    for category, chunks in chunks_data.items():
        print(f"\n--- Processing category: {category} ---")
        chunk_texts = [chunk["chunk"] for chunk in chunks if chunk.get("chunk")][:4]

        output_file = os.path.join(output_dir, f"{category}_KG.json")
        knowledge_graph = create_knowledge_graph(chunk_texts, output_file)
        knowledge_graphs[category] = knowledge_graph
    return knowledge_graphs

# Loads all knowledge graphs from a directory 
def load_all_knowledge_graphs(directory="KG"):
    knowledge_graphs = {}

    for filename in os.listdir(directory):
        if filename.endswith("_KG.json"):
            category = filename.replace("_KG.json", "")
            path = os.path.join(directory, filename)

            with open(path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
                graph = nx.node_link_graph(graph_data)
                knowledge_graphs[category] = graph

    return knowledge_graphs

def plot_and_save_category_knowledge_graph(knowledge_graphs, output_dir="KG", figsize=(20, 15)):
    os.makedirs(output_dir, exist_ok=True)
    for category, graph in knowledge_graphs.items():
       
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph, k=1.0, seed=42)
        nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=1200, alpha=0.9)
        nx.draw_networkx_edges(graph, pos, edge_color='gray', width=1.2, alpha=0.6)
        nx.draw_networkx_labels(
            graph, pos,
            font_size=10,
            font_family='sans-serif',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
        )

        edge_labels = {(u, v): d['relation'] for u, v, d in graph.edges(data=True) if 'relation' in d}
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.4,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.1)
        )

        plt.title(f"Knowledge Graph for Category: {category}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()

        output_file = os.path.join(output_dir, f"{category}_KG.png")
        plt.savefig(output_file, format='png', dpi=300)
        plt.close()

        print(f"Knowledge graph for {category} saved to {output_file}")
