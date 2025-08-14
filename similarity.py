from py2neo import Graph
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Koneksi ke Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

# Ambil data dari Neo4j
query = """
MATCH (p:Paper)
WHERE p.embedding_fastrp IS NOT NULL 
  AND p.embedding_node2vec IS NOT NULL
  AND p.text_embedding IS NOT NULL 
  AND p.groundtruth_embedding IS NOT NULL
RETURN p.id AS id, 
       p.title AS title,
       p.embedding_fastrp AS embedding_fastrp,
       p.embedding_node2vec AS embedding_node2vec,
       p.text_embedding AS text_embedding,
       p.groundtruth_embedding AS groundtruth_embedding
"""
df = graph.run(query).to_data_frame()

# Ubah kolom ke numpy array
for col in ["embedding_fastrp", "embedding_node2vec", "text_embedding", "groundtruth_embedding"]:
    df[col] = df[col].apply(lambda x: np.array(x))

# Ambil ID dan title
ids = df['id'].tolist()
titles = df['title'].tolist()
text_matrix = np.stack(df['text_embedding'].values)

# Fungsi untuk hitung similarity dan simpan JSON
def save_similarity_json(emb_matrix, text_matrix, filename, w1=0.5, w2=0.5):
    """Menghitung similarity gabungan dan simpan ke JSON, urut per source."""
    sim1 = cosine_similarity(emb_matrix)
    sim1 = (sim1 + 1) / 2  # normalisasi ke [0,1]
    sim2 = cosine_similarity(text_matrix)
    sim2 = (sim2 + 1) / 2

    combined_sim = w1 * sim1 + w2 * sim2

    similarity_list = []
    for i, source_title in enumerate(titles):
        # Buat daftar semua target + skor untuk source ini
        target_scores = [
            {"source_title": source_title,
             "target_title": titles[j],
             "score": float(combined_sim[i][j])}
            for j in range(len(titles)) if j != i  # skip diri sendiri kalau mau
        ]
        # Urutkan berdasarkan skor tertinggi
        target_scores.sort(key=lambda x: x["score"], reverse=True)
        similarity_list.extend(target_scores)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(similarity_list, f, indent=2, ensure_ascii=False)
    print(f"✅ Similarity disimpan di '{filename}'")

# Simpan FastRP
save_similarity_json(
    np.stack(df['embedding_fastrp'].values),
    text_matrix,
    "paper-similarity-fastrp.json"
)

# Simpan Node2Vec
save_similarity_json(
    np.stack(df['embedding_node2vec'].values),
    text_matrix,
    "paper-similarity-node2vec.json"
)

# Simpan Groundtruth (hanya 1 embedding)
def save_groundtruth_json(emb_matrix, filename):
    sim = cosine_similarity(emb_matrix)
    sim = (sim + 1) / 2  # normalisasi ke [0,1]
    similarity_list = []
    for i, source_title in enumerate(titles):
        target_scores = [
            {"source_title": source_title,
             "target_title": titles[j],
             "score": float(sim[i][j])}
            for j in range(len(titles)) if j != i
        ]
        target_scores.sort(key=lambda x: x["score"], reverse=True)
        similarity_list.extend(target_scores)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(similarity_list, f, indent=2, ensure_ascii=False)
    print(f"✅ Groundtruth disimpan di '{filename}'")

save_groundtruth_json(
    np.stack(df['groundtruth_embedding'].values),
    "paper-similarity-groundtruth.json"
)
