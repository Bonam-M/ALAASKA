from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging
from typing import Optional, List, Dict
import os

logger = logging.getLogger(__name__)

# Initialize once at module load
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "ds442_vdb")

try:
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
    collection = client.get_collection(name="ds442_hw_sol")
    logger.info("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    embedding_model = None
    collection = None


async def query_homework(query_text: str) -> Optional[List[Dict[str, str]]]:
    """
    Query ChromaDB for top 3 most similar homework questions.
    Returns list of dicts with 'chunk_id' and 'answer_text', or None if failed.
    """
    
    if not collection:
        logger.warning("ChromaDB not available")
        return None
    
    try:
        # Generate embedding and query
        query_embedding = embedding_model.encode(query_text, normalize_embeddings=True, convert_to_numpy=False)
        
        results = collection.query(query_embeddings=[query_embedding.cpu().tolist()], n_results=3  )
        
        if not results.get('ids'):
            logger.warning("No results from ChromaDB")
            return None
        
        # Extract valid results (non-empty answer_text only)
        homework_answers = []
        for i in range(len(results['ids'][0])):
            chunk_id = results['metadatas'][0][i].get('chunk_id', '')
            answer_text = results['metadatas'][0][i].get('answer_text', '')
            
            if answer_text and answer_text.strip():
                homework_answers.append({
                    'chunk_id': chunk_id,
                    'answer_text': answer_text.strip()
                })
                logger.info(f"Found: {chunk_id}, distance: {results['distances'][0][i]:.4f}")
        
        return homework_answers if homework_answers else None
            
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        return None