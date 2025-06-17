import pandas as pd
import json
from pathlib import Path 
from sentence_transformers import SentenceTransformer

def load_data(pos_data_path, neg_data_path):
    pos_data = pd.read_csv(pos_data_path, sep='|', header=None, names=["id", "title", "review", "rating"])
    neg_data = pd.read_csv(neg_data_path, sep='|', header=None, names=["id", "title", "review", "rating"])

    pos_data['label'] = "positive" # used only for evaluation
    neg_data['label'] = "negative" # used only for evaluation

    return pd.concat([pos_data, neg_data], ignore_index=True)

def data_to_json(data_input):
    data = []
    for idx, row in data_input.iterrows():
        data.append({
            "id": row["id"],
            "title": row["title"],
            "review": row["review"],
            "rating": row["rating"],
            "label": row["label"]
        })
    return data

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        for row in data:
            json.dump(row, f)
            f.write('\n')
    print(f"Data saved to {output_path}")

def get_embeddings(data, task:str):
    model = SentenceTransformer("BAAI/bge-base-en-v1.5") # Check if available otherwise use another model
    texts = [f"Represent this sentence for {task}: {row['review']}" for row in data]
    embeddings = model.encode(texts)
    
    for i, row in enumerate(data):
        row['embedding'] = embeddings[i].tolist()
    return data


def main():
    pos_data_path = r"./Datasets/IMDB/train/pos_imdb_data.csv"
    neg_data_path = r"./Datasets/IMDB/train/neg_imdb_data.csv"

    no_embed_path = r"./Datasets/IMDB/currated/imdb_no_embed.json"
    embed_path = r"./Datasets/IMDB/currated/imdb_embed_retrieval.json"

    data_input = load_data(pos_data_path, neg_data_path)
    data_json = data_to_json(data_input)

    save_json(data_json, no_embed_path)

    data_embed = get_embeddings(data_json, task="retrieval")
    save_json(data_embed, embed_path)

    print(f"Saved: {no_embed_path} and {embed_path}")

if __name__ == "__main__":
    main()