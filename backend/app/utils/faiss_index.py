# Functions to save/load FAISS indexes
import faiss
import numpy as np
import os

def save_faiss_index(index, file_path):
    """Save a FAISS index to a file."""
    faiss.write_index(index, file_path)

def load_faiss_index(file_path):
    """Load a FAISS index from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FAISS index file '{file_path}' does not exist.")
    return faiss.read_index(file_path)

# def create_faiss_index(embeddings, dimension=768):
#     """Create and return a FAISS index."""
#     index = faiss.IndexFlatL2(dimension)
#     if not index.is_trained:
#         raise ValueError("FAISS index is not properly trained.")
#     index.add(np.array(embeddings).astype("float32"))
#     return index