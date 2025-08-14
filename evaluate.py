import json
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score, mean_squared_error, mean_absolute_error

# Fungsi untuk membaca JSON
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Fungsi untuk membuat dictionary: source -> [(target, score)]
def to_dict(data):
    sim_dict = {}
    for item in data:
        src = item["source_title"]
        tgt = item["target_title"]
        score = float(item["score"])
        if src not in sim_dict:
            sim_dict[src] = []
        sim_dict[src].append((tgt, score))
    return sim_dict

# Fungsi evaluasi NDCG, Kendall's Tau, MSE, dan MAE
def evaluate(method_dict, groundtruth_dict):
    ndcg_scores = []
    kendall_scores = []
    mse_scores = []
    mae_scores = []

    for src, gt_list in groundtruth_dict.items():
        if src not in method_dict:
            continue

        # Ambil ranking groundtruth
        gt_sorted = sorted(gt_list, key=lambda x: x[1], reverse=True)
        gt_titles = [t for t, _ in gt_sorted]
        gt_scores = [s for _, s in gt_sorted]

        # Ambil ranking dari metode
        method_list = method_dict[src]
        method_sorted = sorted(method_list, key=lambda x: x[1], reverse=True)
        method_titles = [t for t, _ in method_sorted]

        # Samakan urutan untuk perbandingan
        common_titles = list(set(gt_titles) & set(method_titles))
        if len(common_titles) < 2:
            continue

        gt_scores_aligned = [gt_scores[gt_titles.index(t)] for t in common_titles]
        method_scores_aligned = [dict(method_list)[t] for t in common_titles]

        # Hitung NDCG
        ndcg_val = ndcg_score([gt_scores_aligned], [method_scores_aligned])
        ndcg_scores.append(ndcg_val)

        # Hitung Kendall's Tau
        gt_ranks = [gt_titles.index(t) for t in common_titles]
        method_ranks = [method_titles.index(t) for t in common_titles]
        tau, _ = kendalltau(gt_ranks, method_ranks)
        kendall_scores.append(tau)

        # Hitung MSE & MAE berdasarkan skor
        mse_scores.append(mean_squared_error(gt_scores_aligned, method_scores_aligned))
        mae_scores.append(mean_absolute_error(gt_scores_aligned, method_scores_aligned))

    # Rata-rata
    return (
        np.mean(ndcg_scores),
        np.mean(kendall_scores),
        np.mean(mse_scores),
        np.mean(mae_scores)
    )


if __name__ == "__main__":
    # Load data
    node2vec_data = load_json("paper-similarity-node2vec.json")
    fastrp_data = load_json("paper-similarity-fastrp.json")
    groundtruth_data = load_json("paper-similarity-groundtruth.json")

    # Konversi ke dict
    node2vec_dict = to_dict(node2vec_data)
    fastrp_dict = to_dict(fastrp_data)
    groundtruth_dict = to_dict(groundtruth_data)

    # Evaluasi Node2Vec
    ndcg_n2v, tau_n2v, mse_n2v, mae_n2v = evaluate(node2vec_dict, groundtruth_dict)
    print("="*50)
    print("HASIL RATA-RATA - NODE2VEC")
    print("="*50)
    print(f"📊 Rata-rata NDCG        : {ndcg_n2v:.4f}")
    print(f"📊 Rata-rata Kendall's Tau : {tau_n2v:.4f}")
    print(f"📊 Rata-rata MSE         : {mse_n2v:.6f}")
    print(f"📊 Rata-rata MAE         : {mae_n2v:.6f}")

    # Evaluasi FastRP
    ndcg_fastrp, tau_fastrp, mse_fastrp, mae_fastrp = evaluate(fastrp_dict, groundtruth_dict)
    print("="*50)
    print("HASIL RATA-RATA - FASTRP")
    print("="*50)
    print(f"📊 Rata-rata NDCG        : {ndcg_fastrp:.4f}")
    print(f"📊 Rata-rata Kendall's Tau : {tau_fastrp:.4f}")
    print(f"📊 Rata-rata MSE         : {mse_fastrp:.6f}")
    print(f"📊 Rata-rata MAE         : {mae_fastrp:.6f}")
