import os
import chromadb
from sentence_transformers import SentenceTransformer

class VectorStoreRetriever:
    def __init__(self):
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
        if not os.path.exists(db_path):
            print("Vector store not found. Please run build_kb.py first.")
            self.collection = None
        else:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_collection("retention_strategies")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
    def retrieve_strategies(self, query: str, top_k: int = 5) -> list:
        if not self.collection:
            return ["Knowledge base not initialized."]
            
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        return results["documents"][0] if results and "documents" in results else []
