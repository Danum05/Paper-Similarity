import logging
import time
import os
from py2neo import Graph
import numpy as np
import pandas as pd

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graph_embedding")

# Koneksi ke database Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

# Path penyimpanan hasil embedding
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_EMBEDDING_FILE = os.path.join(BASE_DIR, 'Result', 'paper_embeddings.csv')

GRAPH_NAME = 'paperGraph'

def drop_existing_graph():
    """Hapus graph projection yang sudah ada"""
    logger.info(f"Memeriksa graph '{GRAPH_NAME}'...")
    result = graph.run(f"CALL gds.graph.exists('{GRAPH_NAME}') YIELD exists RETURN exists").data()
    if result and result[0]['exists']:
        logger.info(f"Menghapus graph '{GRAPH_NAME}'...")
        graph.run(f"CALL gds.graph.drop('{GRAPH_NAME}', false)")

def create_graph_projection():
    """Buat graph projection dari Paper dengan 2 jenis relasi: CITES dan REFERENCES"""
    logger.info("Membuat graph projection untuk Paper dengan CITES dan REFERENCES...")
    graph.run(f"""
        CALL gds.graph.project(
            '{GRAPH_NAME}',
            'Paper',
            {{
                CITES: {{ type: 'CITES', orientation: 'UNDIRECTED' }},
                REFERENCES: {{ type: 'REFERENCES', orientation: 'UNDIRECTED' }}
            }}
        )
    """)

def run_node2vec():
    """Jalankan node2vec pada graph Paper"""
    logger.info("Menjalankan node2vec untuk Paper...")
    result = graph.run(f"""
        CALL gds.beta.node2vec.stream('{GRAPH_NAME}', {{
            embeddingDimension: 128,
            walkLength: 80,
            returnFactor: 1.0,
            inOutFactor: 1.0
        }})
        YIELD nodeId, embedding
        RETURN gds.util.asNode(nodeId).id AS paperId, embedding
    """).data()
    
    logger.info(f"Node2Vec: Berhasil memproses {len(result)} paper")
    return result

def run_fastrp():
    """Jalankan FastRP pada graph Paper"""
    logger.info("Menjalankan FastRP untuk Paper...")
    result = graph.run(f"""
        CALL gds.fastRP.stream('{GRAPH_NAME}', {{
            embeddingDimension: 128,
            iterationWeights: [0.0, 1.0, 0.0],
            normalizationStrength: 0.0
        }})
        YIELD nodeId, embedding
        RETURN gds.util.asNode(nodeId).id AS paperId, embedding
    """).data()
    
    logger.info(f"FastRP: Berhasil memproses {len(result)} paper")
    return result

def process_embeddings(embeddings, property_name):
    """Normalisasi dan simpan embeddings ke dalam node Paper"""
    valid_embeddings = []
    for record in embeddings:
        try:
            paper_id = record['paperId']
            embedding = np.array(record['embedding'])
            norm = np.linalg.norm(embedding)
            if norm == 0.0:
                logger.warning(f"Embedding nol pada paper {paper_id}, dilewati.")
                continue
            normalized = embedding / norm

            # Simpan ke database
            graph.run(f"""
                MATCH (p:Paper {{id: $id}})
                SET p.{property_name} = $embedding
            """, id=paper_id, embedding=normalized.tolist())

            valid_embeddings.append({
                'paperId': paper_id,
                property_name: normalized.tolist()
            })

        except Exception as e:
            logger.error(f"Error processing {paper_id}: {str(e)}")
    
    return valid_embeddings

def save_to_csv(embeddings_node2vec, embeddings_fastrp):
    """Simpan embeddings ke CSV dengan kedua metode"""
    df_node2vec = pd.DataFrame(embeddings_node2vec)
    df_fastrp = pd.DataFrame(embeddings_fastrp)

    # Gabungkan berdasarkan paperId
    df_merged = pd.merge(df_node2vec, df_fastrp, on='paperId', how='outer')
    df_merged.to_csv(PAPER_EMBEDDING_FILE, index=False)
    logger.info(f"Embeddings Paper (Node2Vec + FastRP) disimpan di {PAPER_EMBEDDING_FILE}")

def main():
    total_start = time.time()
    try:
        drop_existing_graph()
        create_graph_projection()

        # Jalankan Node2Vec
        raw_node2vec = run_node2vec()
        processed_node2vec = process_embeddings(raw_node2vec, 'embedding_node2vec')

        # Jalankan FastRP
        raw_fastrp = run_fastrp()
        processed_fastrp = process_embeddings(raw_fastrp, 'embedding_fastrp')

        # Simpan hasil gabungan ke CSV
        save_to_csv(processed_node2vec, processed_fastrp)

    except Exception as e:
        logger.error(f"Error utama: {str(e)}")
    finally:
        logger.info(f"Total waktu eksekusi: {time.time() - total_start:.2f} detik")

if __name__ == "__main__":
    main()
