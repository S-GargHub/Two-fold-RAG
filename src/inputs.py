
import os
import csv

from datasets import load_dataset
from collections import defaultdict


# Load dataset
def load_data():
    ds = load_dataset("yixuantt/MultiHopRAG", "corpus")
    df = ds["train"].to_pandas()

    ds_q = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
    df_q = ds_q["train"].to_pandas()

    return df, df_q

def prepare_data_by_category(test_data, output_dir="data/subset/inference2/", top_k=25):

    category_urls = defaultdict(set)

    for filename in os.listdir("data/subset/inference/"):
        if not filename.endswith(".csv"):
            continue

        category = filename.replace(".csv", "")
        file_path = os.path.join("data/subset/inference/", filename)

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                urls = row.get("Evidence URLs", "")
                for url in urls.split(";"):
                    url = url.strip()
                    if url:
                        category_urls[category].add(url)

    evidence_urls = set()

    os.makedirs(output_dir, exist_ok=True)

    category_data = defaultdict(lambda: {
        "urls": set(),
        "questions": [],
        "answers": [],
        "rows": []
    })

    filtered = test_data[test_data["question_type"] == "inference_query"]

    for _, row in filtered.iterrows():
        evidence_list = row.get("evidence_list", [])
        if evidence_list is None or len(evidence_list) == 0:
            continue

        # Extract category from first evidence item
        category = evidence_list[0].get("category", "unknown").strip().lower()
        question = row["query"]
        answer = row["answer"]
        if question in category_data[category]["questions"]:
            continue
        evidence_urls = [item.get("url") for item in evidence_list if item.get("url")]
        valid_urls = set(category_urls.get(category, []))
        if any(url not in valid_urls for url in evidence_urls):
            continue
        category_data[category]["questions"].append(question)
        category_data[category]["answers"].append(answer)
        category_data[category]["urls"].update(evidence_urls)

        evidence_str = "; ".join(evidence_urls)
        category_data[category]["rows"].append((question, answer, "inference_query", evidence_str))

    # Write CSV files per category
    for category, data in category_data.items():
        output_file = os.path.join(output_dir, f"{category}.csv")
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Answer", "Type", "Evidence URLs"])
            writer.writerows(data["rows"])
    
    # Return the URLs for the top_k questions in each category
    category_urls = {}
    for category, data in category_data.items():
        category_urls[category] = list(data["urls"])  # Get the list of unique URLs

    return category_data, category_urls

def filter_corpus_by_categories_subset(df_corpus, category_urls):
    unique_categories = df_corpus["category"].unique()
    category_corpus = {category: df_corpus[df_corpus["category"] == category] for category in unique_categories}

    filtered_category_corpus = {}
    for category, df in category_corpus.items():
        valid_urls = category_urls.get(category, [])

        filtered_df = df[df["url"].isin(valid_urls)]
        filtered_category_corpus[category] = filtered_df
        filtered_df.to_csv(f"data/subset/corpus/{category}.csv", index=False)

    return filtered_category_corpus

# Filter the corpus DataFrame to only include documents with URLs in the given list.
def filter_corpus_by_urls(corpus_df, urls):
    filtered_corpus = corpus_df[corpus_df["url"].isin(urls)].reset_index(drop=True)
    return filtered_corpus

def load_qna_from_category_files(input_dir="data/subset/inference2/"):
    qna_by_category = {}

    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return qna_by_category

    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for file in files:
        category = file.replace(".csv", "")
        file_path = os.path.join(input_dir, file)

        qna_by_category[category] = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row.get("Question", "").strip()
                answer = row.get("Answer", "").strip()
                if question and answer:
                    qna_by_category[category].append((question, answer))

    return qna_by_category
