# High-level qa_system_with_individual_summaries
import os
import pickle
import json
from sentence_transformers import SentenceTransformer
from app.services.pdf_processing import extract_text_from_pdf
from app.services.chunking import agentic_chunking
from app.services.keyword_extraction import extract_keywords_keybert
from app.services.embedding_service import (create_embeddings, 
                                            retrieve_relevant_chunks1,
                                            create_faiss_index,
                                            assign_chunk_ids)
from app.services.summarization import generate_with_flant5_individual
#need to divide into query ands upoading
# def qa_system_with_individual_summaries(pdf_path, query, chunk_size=300, overlap=50, top_k=1):
#     """
#     QA system with independent summarization of retrieved chunks.

#     Args:
#         pdf_path (str): Path to the PDF document.
#         query (str): User's query.
#         chunk_size (int): Maximum token size for each chunk.
#         overlap (int): Overlap size between consecutive chunks.
#         top_k (int): Number of relevant chunks to retrieve.

#     Returns:
#         str: Final aggregated summaries of retrieved chunks.
#     """
#     # Step 1: Extract and chunk text from the PDF
#     text = extract_text_from_pdf(pdf_path)
#     chunks = agentic_chunking(text, max_tokens=chunk_size, overlap=overlap)

#     # Step 2: Embed chunks and create FAISS index
#     model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     flat_chunks, chunk_ids = assign_chunk_ids([chunks])
#     # Generate embeddings
#     embeddings = create_embeddings(flat_chunks)
#     index = create_faiss_index(embeddings)

#     # Step 3: Retrieve relevant chunks

#     #relevant_chunks = retrieve_relevant_chunks(query, index, chunks, model, top_k=top_k)
#     relevant_chunks = retrieve_relevant_chunks1(query, index, flat_chunks, chunk_ids, model)
#     print(f"Retrieved {relevant_chunks}")

#     # Step 4: Summarize each chunk independently model_name="google/t5-large-lm-adapt"

#     final_answer = generate_with_flant5_individual(relevant_chunks, query) #this only good but trying down
#     #final_answer=generate_with_distilbart(relevant_chunks, query)

#     return final_answer

#new ones need to check if they mesh    
# Load and prepare the FAISS index
# pdf_path = "path_to_your_pdf.pdf"
# index, flat_chunks, chunk_ids, model = load_pdf_to_faiss(pdf_path)

# # Use the QA system to retrieve and summarize relevant chunks
# query = "Your query here"
# result = qa_system_with_individual_summaries_faiss(index, flat_chunks, chunk_ids, model, query, top_k=3)

# print("Final Answer:", result)
import faiss
def retrieve_answer(file_name, query, top_k=3):
    """
    Retrieve an answer using the preloaded index and data for the specific file.

    Args:
        file_name (str): Unique name to identify the saved data.
        query (str): Query string for the QA system.
        top_k (int): Number of top relevant chunks to retrieve.

    Returns:
        str: Final summarized response.
    """
    base_path = "saved_files"
    index_path = os.path.join(base_path, f"{file_name}_index.faiss")
    metadata_path = os.path.join(base_path, f"{file_name}_metadata.pkl")
    
    # Load FAISS index
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No data found for file: {file_name}")

    index = faiss.read_index(index_path)

    # Load metadata
    with open(metadata_path, "rb") as meta_file:
        metadata = pickle.load(meta_file)
    flat_chunks = metadata['flat_chunks']
    chunk_ids = metadata['chunk_ids']
    model_name = metadata['model_name']
    model = SentenceTransformer(model_name)

    # Retrieve and summarize
    relevant_chunks = retrieve_relevant_chunks1(query, index, flat_chunks, chunk_ids, model)
    final_answer = generate_with_flant5_individual(relevant_chunks, query)

    return final_answer

def qa_system_with_individual_summaries_faiss(index, flat_chunks, chunk_ids, model, query, top_k=1):
    """
    Performs QA with the FAISS index and independent summarization of retrieved chunks.

    Args:
        index: FAISS index for chunk retrieval.
        flat_chunks (list): List of all flat chunks.
        chunk_ids (list): List of chunk IDs.
        model: Embedding model for query embeddings.
        query (str): User's query.
        top_k (int): Number of relevant chunks to retrieve.

    Returns:
        str: Final aggregated summaries of retrieved chunks.
    """
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks1(query, index, flat_chunks, chunk_ids, model)
    
    print(relevant_chunks)
    return 'bruhhh'

    print(f"Retrieved {len(relevant_chunks)} chunks: {relevant_chunks}")

    # Summarize each chunk independently
    final_answer = generate_with_flant5_individual(relevant_chunks, query)  
    # Alternate summarization
    # final_answer = generate_with_distilbart(relevant_chunks, query)

    return final_answer
#check below
def display_chunks(file_name, top_k=1):
    """
    Retrieve an answer using the preloaded index and data for the specific file.

    Args:
        file_name (str): Unique name to identify the saved data.
        query (str): Query string for the QA system.
        top_k (int): Number of top relevant chunks to retrieve.

    Returns:
        str: Final summarized response.
    """
    base_path = "saved_files"
    index_path = os.path.join(base_path, f"{file_name}_index.faiss")
    metadata_path = os.path.join(base_path, f"{file_name}_metadata.pkl")
    
    # Load FAISS index
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No data found for file: {file_name}")

    index = faiss.read_index(index_path)

    # Load metadata
    with open(metadata_path, "rb") as meta_file:
        metadata = pickle.load(meta_file)
        flat_chunks = metadata['flat_chunks']
        chunk_ids = metadata['chunk_ids']
        model_name = metadata['model_name']
        model = SentenceTransformer(model_name)
    chunks_with_key_words=[]

    for i in flat_chunks:
        out=extract_keywords_keybert(i)
        chunks_with_key_words.append([i,out])

    # Retrieve and summarize
    

    return chunks_with_key_words