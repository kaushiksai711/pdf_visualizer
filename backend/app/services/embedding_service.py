# create_embeddings, create_faiss_index, retrieval-related functions
from sentence_transformers import SentenceTransformer
import numpy as np
from app.services.pdf_processing import extract_text_from_pdf
from app.services.chunking import agentic_chunking

# Initialize the model globally
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Converts the text chunks into embeddings using a pre-trained SentenceTransformer model.

    Args:
        chunks (list): List of text chunks to be embedded.
        model_name (str): The model name used to generate embeddings.

    Returns:
        numpy.ndarray: Embeddings for the input chunks.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)  # Generate embeddings for the chunks
    return embeddings

import faiss

def create_faiss_index(embeddings):
    """
    Creates a FAISS index to allow efficient similarity search.

    Args:
        embeddings (numpy.ndarray): The array of embeddings generated for the chunks.

    Returns:
        faiss.IndexFlatL2: The FAISS index object.
    """
    # Create an index for L2 (Euclidean) distance search
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))  # Add the embeddings to the index
    return index


import os
import pickle
import json
from sentence_transformers import SentenceTransformer

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

def load_pdf_to_faiss(file_name, file_path, chunk_size=300, overlap=50):
    """Load PDF and create FAISS index."""
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        print(text,'text')
        # Create chunks
        chunks = agentic_chunking(text, max_tokens=chunk_size, overlap=overlap)
        
        # Assign chunk IDs
        flat_chunks, chunk_ids = assign_chunk_ids([chunks])  # Wrap chunks in list since it's a single document
        # Create embeddings
        embeddings = create_embeddings(flat_chunks)
        
        # Create FAISS index
        index = create_faiss_index(embeddings)
        
        # Initialize model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Processed {len(flat_chunks)} chunks with IDs: {chunk_ids[:5]}...")
        
        return index, flat_chunks, chunk_ids, model
        
    except Exception as e:
        print(f"Error in load_pdf_to_faiss: {str(e)}")
        print(f"File path: {file_path}")
        print(f"Number of chunks: {len(chunks) if 'chunks' in locals() else 'N/A'}")
        raise

def assign_chunk_ids(chunks):
    """
    Assigns a unique ID to each chunk in the format doc_[doc_number]_[chunk_number].

    Args:
        chunks (list): A nested list of chunks for all documents (list of lists).

    Returns:
        list, list: Flattened list of chunks with assigned IDs and a list of IDs.
    """
    flat_chunks = []
    chunk_ids = []

    for doc_idx, doc_chunks in enumerate(chunks):
        for chunk_idx, chunk in enumerate(doc_chunks):
            chunk_id = f"doc_{doc_idx + 1}_{chunk_idx + 1}"
            flat_chunks.append(chunk)
            chunk_ids.append(chunk_id)

    return flat_chunks, chunk_ids
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def retrieve_relevant_chunks1(query, index, num_chunks=3):
    """
    Retrieve relevant chunks using FAISS index.
    
    Args:
        query (str): The query text
        index: FAISS index
        num_chunks (int): Number of chunks to retrieve (default: 3)
    
    Returns:
        list: List of relevant chunk indices
    """
    try:
        # Encode the query
        query_embedding = model.encode([query])
        
        # Search in FAISS index
        D, I = index.search(query_embedding, num_chunks)
        
        # Convert numpy array to list of Python integers and ensure indices are valid
        indices = [int(idx) for idx in I[0] if idx >= 0]
        
        print(f"Retrieved indices: {indices}")  # Debug log
        return indices
    except Exception as e:
        print(f"Error in retrieve_relevant_chunks1: {str(e)}")
        print(f"Query embedding shape: {query_embedding.shape if 'query_embedding' in locals() else 'N/A'}")
        print(f"FAISS search results - D: {D if 'D' in locals() else 'N/A'}, I: {I if 'I' in locals() else 'N/A'}")
        raise

# def retrieve_relevant_chunks1(query, index, chunks, chunk_ids, model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"), top_k=20):
#     """
#     Retrieves the most relevant text chunks for a given query using the FAISS index.

#     Args:
#         query (str): The user's query.
#         index (faiss.IndexFlatL2): The FAISS index of embeddings.
#         chunks (list): List of original text chunks corresponding to the embeddings.
#         chunk_ids (list): List of IDs corresponding to the chunks.
#         model (SentenceTransformer): The embedding model used for encoding.
#         top_k (int): The number of most relevant chunks to retrieve.

#     Returns:
#         list of tuples: The most relevant text chunks with their IDs (chunk_id, chunk).
#     """
#     # Generate embedding for the query
#     query_embedding = model.encode([query])  # Embedding for the query
#     print('aasadaasdasd')
#     # Search the index for top_k most similar chunks
#     distances, indices = index.search(query_embedding, top_k)
#     relevant_chunks=[]
#     uniques=[]
#     for i, dist in zip(indices[0], distances[0]):
#       if dist>=0.75 and chunk_ids[i] not in uniques:
#         relevant_chunks.append([chunk_ids[i], chunks[i]])
#         uniques.append(chunk_ids[i])
#     # Retrieve the relevant chunks and their IDs
#     # relevant_chunks = [
#     #     (chunk_ids[i], chunks[i]) for i, dist in zip(indices[0], distances[0]) if dist >= 0.75
#     # ]

#     return relevant_chunks
# def retrieve_relevant_chunks(query, index, chunks, model, top_k=20):
#     """
#     Retrieves the most relevant text chunks for a given query using the FAISS index.

#     Args:
#         query (str): The user's query.
#         index (faiss.IndexFlatL2): The FAISS index of embeddings.
#         chunks (list): List of original text chunks corresponding to the embeddings.
#         model (SentenceTransformer): The embedding model used for encoding.
#         top_k (int): The number of most relevant chunks to retrieve.

#     Returns:
#         list: The most relevant text chunks.
#     """
#     # Generate embedding for the query
#     query_embedding = model.encode([query])  # Embedding for the query

#     # Search the index for top_k most similar chunks
#     distances, indices = index.search(query_embedding, top_k)

#     # Retrieve the relevant chunks
#     relevant_chunks = [chunks[i] for i, dist in zip(indices[0], distances[0]) if dist >= 0.75]

#     return relevant_chunks

def retrieve_chunk_by_id(chunk_id, chunks, chunk_ids):
    """
    Retrieves a specific chunk by its chunk ID.

    Args:
        chunk_id (str): The ID of the chunk to retrieve.
        chunks (list): List of all text chunks.
        chunk_ids (list): List of IDs corresponding to the chunks.

    Returns:
        str: The chunk text associated with the given chunk ID.
    """
    try:
        # Find the index of the chunk ID
        index = chunk_ids.index(chunk_id)
        return chunks[index]
    except ValueError:
        return f"Chunk with ID '{chunk_id}' not found."
