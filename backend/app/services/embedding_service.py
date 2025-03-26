# create_embeddings, create_faiss_index, retrieval-related functions
from sentence_transformers import SentenceTransformer
import numpy as np
from app.services.pdf_processing import extract_text_from_pdf
from app.services.chunking import agentic_chunking
from app.services.summarization import generate_comprehensive_response
from typing import List
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
        # Create chunks
        chunks = agentic_chunking(text, max_tokens=chunk_size, overlap=overlap)
        
        # Assign chunk IDs
        flat_chunks, chunk_ids = assign_chunk_ids(file_name,[chunks])  # Wrap chunks in list since it's a single document
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

def assign_chunk_ids(file_name,chunks):
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
            chunk_id = f"doc_{file_name}_{doc_idx + 1}_{chunk_idx + 1}"
            flat_chunks.append(chunk)
            chunk_ids.append(chunk_id)
    print(chunk_ids)

    return flat_chunks, chunk_ids
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
def retrieve_comprehensive_answer(file_names, query, top_k=3):
    """
    Retrieve a comprehensive answer using the preloaded indices from multiple files.
    
    Args:
        file_names (list): List of file names to process.
        query (str): Query string for the QA system.
        top_k (int): Number of top relevant chunks per file.
    
    Returns:
        str: Final comprehensive response.
    """
    base_path = "saved_files"
    all_relevant_chunks = []
    print('retrieveeeeee',file_names)
    
    # Collect relevant chunks from all files
    for file_name in file_names:
        index_path = os.path.join(base_path, f"{file_name}_index.faiss")
        metadata_path = os.path.join(base_path, f"{file_name}_metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No data found for file: {file_name}")

        # Load FAISS index and metadata
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as meta_file:
            metadata = pickle.load(meta_file)
        
        flat_chunks = metadata['flat_chunks']
        chunk_ids = metadata['chunk_ids']
        model_name = metadata['model_name']
        model = SentenceTransformer(model_name)
        print(chunk_ids,'addsdadassss')
        
        # Get relevant chunks from this file
        file_chunks = retrieve_relevant_chunks1(query, index, flat_chunks, chunk_ids, model)
        
        # Add source file information to chunks
        for chunk in file_chunks:
            if isinstance(chunk, str):
                chunk_with_source = {
                    'content': chunk,
                    'source': file_name
                }
                all_relevant_chunks.append(chunk_with_source)
            elif isinstance(chunk, dict):
                chunk['source'] = file_name
                all_relevant_chunks.append(chunk)

    # Generate comprehensive response
    print(all_relevant_chunks,'asxandasndkanssa')
    final_answer = generate_comprehensive_response(all_relevant_chunks, query)
    return final_answer



def retrieve_relevant_chunks1(
    query: str,
    index: faiss.Index,
    flat_chunks: List[str],
    chunk_ids: List[int],
    model: SentenceTransformer,
    num_chunks: int = 3,
    similarity_threshold: float = 0.3,
    max_retries: int = 2,
    batch_size: int = 32
) -> List[dict]:
    """
    Retrieve relevant chunks using FAISS index with improved error handling and scoring.
    
    Args:
        query (str): The query text
        index (faiss.Index): FAISS index for similarity search
        flat_chunks (List[str]): List of original text chunks
        chunk_ids (List[int]): List of chunk identifiers
        model (SentenceTransformer): Sentence transformer model for encoding
        num_chunks (int): Number of chunks to retrieve (default: 3)
        similarity_threshold (float): Minimum similarity score to include chunk (default: 0.3)
        max_retries (int): Maximum number of retries for failed encoding (default: 2)
        batch_size (int): Batch size for encoding (default: 32)
    
    Returns:
        List[dict]: List of dictionaries containing chunks and their metadata
    """
    def normalize_query(query: str) -> str:
        """Normalize query text by removing extra whitespace and lowercasing."""
        return " ".join(query.lower().split())
    
    def encode_with_retry(text: str, retry_count: int = 0) -> np.ndarray:
        """Encode text with retry mechanism for robustness."""
        try:
            # Add error handling for empty or invalid text
            if not text or not isinstance(text, str):
                raise ValueError(f"Invalid text input: {text}")
                    
            encoding = model.encode([text], batch_size=batch_size)
            return encoding
            
        except Exception as e:
            if retry_count < max_retries:
                print(f"Retry {retry_count + 1} after encoding error: {str(e)}")
                time.sleep(1)  # Add delay between retries
                return encode_with_retry(text, retry_count + 1)
            raise Exception(f"Failed to encode after {max_retries} retries: {str(e)}")

    def compute_chunk_scores(distances: np.ndarray) -> List[float]:
        """Convert FAISS distances to similarity scores."""
        # FAISS returns L2 distances, convert to similarity scores
        max_distance = np.max(distances) + 1e-6  # Avoid division by zero
        similarities = 1 - (distances / max_distance)
        return similarities.tolist()

    try:
        # Input validation
        if not query:
            raise ValueError("Empty query provided")
        if not isinstance(index, faiss.Index):
            raise TypeError("Invalid FAISS index provided")
        if len(flat_chunks) != len(chunk_ids):
            raise ValueError("Mismatch between chunks and chunk IDs")

        # Normalize and encode query
        normalized_query = normalize_query(query)
        query_embedding = encode_with_retry(normalized_query)

        # Perform FAISS search
        distances, indices = index.search(query_embedding, min(num_chunks, len(flat_chunks)))
        
        # Convert to similarity scores
        similarity_scores = compute_chunk_scores(distances[0])
        
        # Create result list with metadata
        results = []
        for idx, (faiss_idx, score) in enumerate(zip(indices[0], similarity_scores)):
            if faiss_idx >= 0 and score <= similarity_threshold:  # Valid index and meets threshold
                chunk_info = {
                    'content': flat_chunks[faiss_idx],
                    'chunk_id': chunk_ids[faiss_idx],
                    'similarity_score': float(score),
                    'rank': idx + 1
                }
                results.append(chunk_info)

        # Sort by similarity score in descending order
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Log retrieval statistics
        print(f"Query: '{query}'")
        print(f"Retrieved {len(results)} chunks above threshold {similarity_threshold}")
        print(f"Score range: {min(similarity_scores):.3f} - {max(similarity_scores):.3f}")

        return results

    except Exception as e:
        error_context = {
            'query': query,
            'query_length': len(query) if isinstance(query, str) else None,
            'index_size': index.ntotal if hasattr(index, 'ntotal') else None,
            'num_chunks_requested': num_chunks,
            'error_type': type(e).__name__,
            'error_message': str(e)
        }
        print(f"Error in retrieve_relevant_chunks1: {json.dumps(error_context, indent=2)}")
        raise

def process_retrieved_chunks(retrieved_chunks: List[dict], 
                           context_window: int = 100) -> List[dict]:
    """
    Post-process retrieved chunks by adding context and highlighting relevant parts.

    Args:
        retrieved_chunks (List[dict]): List of retrieved chunks with metadata
        context_window (int): Number of characters to include before/after for context

    Returns:
        List[dict]: Processed chunks with added context and highlights
    """
    processed_chunks = []
    
    for chunk in retrieved_chunks:
        content = chunk['content']
        
        # Add surrounding context if available
        start_idx = max(0, chunk.get('start_idx', 0) - context_window)
        end_idx = min(len(content), chunk.get('end_idx', len(content)) + context_window)
        
        processed_chunk = {
            **chunk,
            'content_with_context': content[start_idx:end_idx],
            'context_start_idx': start_idx,
            'context_end_idx': end_idx,
            'has_additional_context': start_idx > 0 or end_idx < len(content)
        }
        processed_chunks.append(processed_chunk)
    
    return processed_chunks
# def retrieve_relevant_chunks1(query, index, num_chunks=3):
#     """
#     Retrieve relevant chunks using FAISS index.
    
#     Args:
#         query (str): The query text
#         index: FAISS index
#         num_chunks (int): Number of chunks to retrieve (default: 3)
    
#     Returns:
#         list: List of relevant chunk indices
#     """
#     try:
#         # Encode the query
#         query_embedding = model.encode([query])
        
#         # Search in FAISS index
#         D, I = index.search(query_embedding, num_chunks)
        
#         # Convert numpy array to list of Python integers and ensure indices are valid
#         indices = [int(idx) for idx in I[0] if idx >= 0]
        
#         print(f"Retrieved indices: {indices}")  # Debug log
#         return indices
#     except Exception as e:
#         print(f"Error in retrieve_relevant_chunks1: {str(e)}")
#         print(f"Query embedding shape: {query_embedding.shape if 'query_embedding' in locals() else 'N/A'}")
#         print(f"FAISS search results - D: {D if 'D' in locals() else 'N/A'}, I: {I if 'I' in locals() else 'N/A'}")
#         raise

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
