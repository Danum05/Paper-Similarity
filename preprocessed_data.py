import re
import json
import logging
import pandas as pd
from py2neo import Graph
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import nltk
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download stopwords if not already present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load sentence transformer model
logger.info("Loading sentence transformer model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Model loaded.")

# Connect to Neo4j
try:
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))
    graph.run("RETURN 1")
    logger.info("Successfully connected to Neo4j.")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    exit()

# === Text Preprocessing ===
def case_folding(text):
    text = text.lower()
    text = re.sub(r"[^\w\s\.\,\!\?\']", "", text)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text):
    if not text or pd.isna(text):
        return ""
    return case_folding(text)

def preprocess_and_chunk(title, abstract, max_chunk_len=1000):
    combined = preprocess_text(f"{title} {abstract}")
    return [combined[i:i+max_chunk_len] for i in range(0, len(combined), max_chunk_len)]

def average_chunk_embeddings(chunks):
    if not chunks or not chunks[0]:
        return []
    embeddings = model.encode(chunks)
    return embeddings.mean(axis=0).tolist()

# === Utility Functions ===
def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_dataframe(data):
    clean_data = []
    for i, item in enumerate(data):
        try:
            if all(k in item for k in ("paperId", "title", "abstract")):
                groundtruth_embedding = []
                if 'embedding' in item and isinstance(item['embedding'], str):
                    try:
                        groundtruth_embedding = json.loads(item['embedding'])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse 'embedding' for paperId {item['paperId']}.")
                
                clean_item = {
                    'id': item.get('paperId'),
                    'title': item.get('title'),
                    'abstract': item.get('abstract'),
                    'year': item.get('year', None),
                    'groundtruth_embedding': groundtruth_embedding,
                    'citationIn': item.get('citationIn', []),
                    'citationOut': item.get('citationOut', [])
                }
                clean_data.append(clean_item)
            else:
                logger.warning(f"Skipping invalid entry at index {i}: missing paperId, title, or abstract.")
        except Exception as e:
            logger.error(f"Error processing item at index {i}: {e}")

    if not clean_data:
        raise ValueError("No valid entries found in the dataset.")
    
    return pd.DataFrame.from_dict(clean_data)

def get_all_citation_ids(df):
    """Get all unique paper IDs mentioned in citations"""
    all_citation_ids = set()
    for citations in df['citationIn']:
        all_citation_ids.update(citations)
    for citations in df['citationOut']:
        all_citation_ids.update(citations)
    all_citation_ids.discard('')
    all_citation_ids.discard(None)
    return list(all_citation_ids)

def categorize_papers(df):
    """Categorize papers into main dataset and shadow nodes"""
    main_paper_ids = set(df['id'].tolist())
    all_citation_ids = set(get_all_citation_ids(df))
    
    shadow_paper_ids = all_citation_ids - main_paper_ids
    
    logger.info(f"Main papers in dataset: {len(main_paper_ids)}")
    logger.info(f"Total citation IDs: {len(all_citation_ids)}")
    logger.info(f"Shadow nodes needed: {len(shadow_paper_ids)}")
    
    return main_paper_ids, shadow_paper_ids

# === Neo4j Insertion (MODIFIED SECTION) ===

def drop_old_constraints():
    logger.info("Dropping old constraint on 'Paper(title)' if it exists...")
    try:
        graph.run("DROP CONSTRAINT paper_title IF EXISTS")
        logger.info("Constraint 'paper_title' dropped successfully.")
    except Exception as e:
        logger.warning(f"Could not drop constraint 'paper_title' (it might not exist or have a different name): {e}")

def create_constraints():
    graph.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
    logger.info("Constraint 'paper_id' created or already exists.")

def create_main_paper_nodes(df):
    """Create main paper nodes with full details"""
    data = df[["id", "title", "abstract", "year"]].to_dict('records')
    
    query = """
    UNWIND $data AS row
    MERGE (p:Paper {id: row.id})
    SET p.title = row.title,
        p.abstract = row.abstract,
        p.year = row.year,
        p.is_main = true,
        p.has_full_data = true
    """
    graph.run(query, data=data)
    logger.info(f"Created/updated {len(data)} main paper nodes with full details.")

def create_shadow_paper_nodes(shadow_ids):
    """Create shadow paper nodes without full details"""
    if not shadow_ids:
        logger.info("No shadow nodes to create.")
        return
    
    shadow_data = [{"id": shadow_id} for shadow_id in shadow_ids]
    
    query = """
    UNWIND $data AS row
    MERGE (p:Paper {id: row.id})
    SET p.is_main = false,
        p.has_full_data = false,
        p.title = COALESCE(p.title, '[Shadow Node - No Title]'),
        p.abstract = COALESCE(p.abstract, '[Shadow Node - No Abstract]')
    """
    graph.run(query, data=shadow_data)
    logger.info(f"Created/updated {len(shadow_data)} shadow paper nodes.")

def update_paper_embeddings(df):
    """Update embeddings for main papers only"""
    logger.info("Generating new text embeddings from title and abstract...")
    df["text_chunks"] = df.apply(lambda row: preprocess_and_chunk(row["title"], row["abstract"]), axis=1)
    df["text_embedding"] = df["text_chunks"].apply(average_chunk_embeddings)
    
    data = df[["id", "text_embedding", "groundtruth_embedding"]].to_dict('records')
    query = """
    UNWIND $data AS row
    MATCH (p:Paper {id: row.id})
    WHERE p.is_main = true
    SET p.text_embedding = row.text_embedding,
        p.groundtruth_embedding = row.groundtruth_embedding
    """
    graph.run(query, data=data)
    logger.info(f"Updated embeddings for {len(data)} main paper nodes.")

def insert_citation_relationships(df):
    """Create citation relationships with proper handling of main and shadow nodes"""
    cites_relationships = []
    cited_by_relationships = []
    
    for _, row in df.iterrows():
        paper_id = row['id']
        
        # Process citationOut (this paper CITES others)
        for cited_id in row['citationOut']:
            if paper_id and cited_id:
                cites_relationships.append({
                    "citing_id": paper_id, 
                    "cited_id": cited_id
                })
        
        # Process citationIn (others CITE this paper)
        for citing_id in row['citationIn']:
            if citing_id and paper_id:
                cited_by_relationships.append({
                    "citing_id": citing_id, 
                    "cited_id": paper_id
                })
    
    # Insert CITES relationships
    if cites_relationships:
        query = """
        UNWIND $data AS row
        MATCH (citing:Paper {id: row.citing_id})
        MATCH (cited:Paper {id: row.cited_id})
        MERGE (citing)-[:CITES]->(cited)
        """
        graph.run(query, data=cites_relationships)
        logger.info(f"Created {len(cites_relationships)} [:CITES] relationships from citationOut data.")
    
    # Insert CITED_BY relationships
    if cited_by_relationships:
        query = """
        UNWIND $data AS row
        MATCH (citing:Paper {id: row.citing_id})
        MATCH (cited:Paper {id: row.cited_id})
        MERGE (citing)-[:CITED_BY]->(cited)
        """
        graph.run(query, data=cited_by_relationships)
        logger.info(f"Created {len(cited_by_relationships)} [:CITED_BY] relationships from citationIn data.")

def count_nodes_and_relationships():
    """Counts nodes and relationships with detailed breakdown"""
    node_queries = {
        "Total Paper Nodes": "MATCH (n:Paper) RETURN count(n) AS count",
        "Main Papers (with full data)": "MATCH (n:Paper) WHERE n.is_main = true RETURN count(n) AS count",
        "Shadow Papers (citations only)": "MATCH (n:Paper) WHERE n.is_main = false RETURN count(n) AS count"
    }
    
    rel_queries = {
        "CITES": "MATCH ()-[r:CITES]->() RETURN count(r) AS count",
        "CITED_BY": "MATCH ()-[r:CITED_BY]->() RETURN count(r) AS count"
    }

    logger.info("\n=== Graph Stats ===")
    for label, query in node_queries.items():
        try:
            result = graph.run(query).data()
            count = result[0]['count'] if result else 0
            logger.info(f"{label}: {count}")
        except Exception as e:
            logger.error(f"Could not count {label}: {e}")

    for rel, query in rel_queries.items():
        try:
            result = graph.run(query).data()
            count = result[0]['count'] if result else 0
            logger.info(f"'{rel}' Relationships: {count}")
        except Exception as e:
            logger.error(f"Could not count {rel}: {e}")

def analyze_citation_network():
    """Analysis with distinction between main and shadow papers"""
    queries = {
        "Most cited MAIN papers": """
            MATCH (p:Paper)<-[:CITES]-(:Paper)
            WHERE p.is_main = true AND p.title IS NOT NULL AND p.title <> '[Shadow Node - No Title]'
            RETURN p.title AS title, count(*) AS count
            ORDER BY count DESC
            LIMIT 10
        """,
        "Most citing MAIN papers": """
            MATCH (p:Paper)-[:CITES]->(:Paper)
            WHERE p.is_main = true AND p.title IS NOT NULL AND p.title <> '[Shadow Node - No Title]'
            RETURN p.title AS title, count(*) AS count
            ORDER BY count DESC
            LIMIT 10
        """,
        "Most cited SHADOW papers (by ID)": """
            MATCH (p:Paper)<-[:CITES]-(:Paper)
            WHERE p.is_main = false
            RETURN p.id AS paper_id, count(*) AS count
            ORDER BY count DESC
            LIMIT 10
        """,
        "Citation patterns": """
            MATCH (main:Paper)-[:CITES]->(cited:Paper)
            WHERE main.is_main = true
            RETURN 
                cited.is_main AS cited_is_main,
                count(*) AS count
            ORDER BY cited_is_main DESC
        """
    }

    logger.info("\n=== Citation Network Analysis ===")
    for label, cypher in queries.items():
        try:
            result = graph.run(cypher).data()
            logger.info(f"\n--- {label} ---")
            if not result:
                logger.info("   No results found.")
            else:
                for row in result:
                    if 'title' in row:
                        title = row.get("title", "[UNKNOWN TITLE]")
                        count = row.get("count", 0)
                        logger.info(f"   {count:<5} | {title}")
                    elif 'paper_id' in row:
                        paper_id = row.get("paper_id", "[UNKNOWN ID]")
                        count = row.get("count", 0)
                        logger.info(f"   {count:<5} | {paper_id}")
                    elif 'cited_is_main' in row:
                        is_main = row.get("cited_is_main")
                        count = row.get("count", 0)
                        target_type = "Main Papers" if is_main else "Shadow Papers"
                        logger.info(f"   Citations to {target_type}: {count}")
        except Exception as e:
            logger.error(f"Error running analysis for '{label}': {e}")

def verify_citation_integrity():
    """Verify that all citation relationships have corresponding nodes"""
    verification_queries = {
        "Broken CITES relationships (missing nodes)": """
            MATCH (citing:Paper)-[r:CITES]->(cited:Paper)
            WHERE citing.id IS NULL OR cited.id IS NULL
            RETURN count(r) AS broken_count
        """,
        "CITES relationships summary": """
            MATCH (citing:Paper)-[:CITES]->(cited:Paper)
            RETURN 
                citing.is_main AS citing_is_main,
                cited.is_main AS cited_is_main,
                count(*) AS relationship_count
            ORDER BY citing_is_main DESC, cited_is_main DESC
        """
    }
    
    logger.info("\n=== Citation Integrity Check ===")
    for label, query in verification_queries.items():
        try:
            result = graph.run(query).data()
            logger.info(f"\n--- {label} ---")
            for row in result:
                if 'broken_count' in row:
                    count = row.get('broken_count', 0)
                    if count == 0:
                        logger.info("   ✅ No broken relationships found.")
                    else:
                        logger.info(f"   ❌ Found {count} broken relationships!")
                elif 'relationship_count' in row:
                    citing_main = row.get('citing_is_main')
                    cited_main = row.get('cited_is_main')
                    count = row.get('relationship_count', 0)
                    citing_type = "Main" if citing_main else "Shadow"
                    cited_type = "Main" if cited_main else "Shadow"
                    logger.info(f"   {citing_type} -> {cited_type}: {count}")
        except Exception as e:
            logger.error(f"Error running verification for '{label}': {e}")

# === Main Pipeline (MODIFIED) ===

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    papers_path = os.path.join(base_dir, 'Dataset', 'dataset.json')
    save_path = os.path.join(base_dir, 'Result', 'processed_paper_data.json')

    # 1. Load and Process Data
    logger.info(f"Loading data from {papers_path}")
    data_papers = load_data_from_json(papers_path)
    df_papers = create_dataframe(data_papers)
    logger.info(f"Created DataFrame with {len(df_papers)} papers.")

    # 2. Categorize papers
    main_paper_ids, shadow_paper_ids = categorize_papers(df_papers)

    # 3. Setup Neo4j Schema and Create Nodes
    drop_old_constraints()
    create_constraints()
    
    # Create main paper nodes with full data
    create_main_paper_nodes(df_papers)
    
    # Create shadow paper nodes
    create_shadow_paper_nodes(list(shadow_paper_ids))
    
    # Update embeddings for main papers only
    update_paper_embeddings(df_papers)

    # 4. Insert Citation Relationships
    insert_citation_relationships(df_papers)

    # 5. Save Processed Data locally
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_save = df_papers.drop(columns=['text_chunks'], errors='ignore')
    df_save.to_json(save_path, orient='records', force_ascii=False, indent=4)
    logger.info(f"Saved processed data to {save_path}")

    # 6. Analyze and Verify the graph
    count_nodes_and_relationships()
    analyze_citation_network()
    verify_citation_integrity()

if __name__ == "__main__":
    main()