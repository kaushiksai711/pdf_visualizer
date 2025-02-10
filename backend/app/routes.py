# API route definitions
# routes.py
import os
from flask import Blueprint, request, jsonify
from app.services.pdf_processing import extract_text_from_pdf
from app.services.chunking import agentic_chunking
from app.services.embedding_service import (load_pdf_to_faiss, 
                                            retrieve_relevant_chunks1)
from app.services.keyword_extraction import extract_keywords_keybert
from app.services.query_handling import  retrieve_answer ,qa_system_with_individual_summaries_faiss, display_chunks
from app.services.chatbot_service import ChatbotManager
from app.services.summarization import generate_with_flant5_individual
import uuid
from app.services.knowledge_graph import generate_knowledge_graph
from werkzeug.utils import secure_filename

api_blueprint = Blueprint("api", __name__)

# Define and create uploads folder if it doesn't exist
UPLOADS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

TEMP_FOLDER = os.path.join(UPLOADS_FOLDER, "temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Initialize the chatbot manager
chatbot_manager = ChatbotManager()

@api_blueprint.route("/upload", methods=["POST"])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    # Ensure uploads directory exists
    os.makedirs(UPLOADS_FOLDER, exist_ok=True)  # Double-check directory exists
    
    files = request.files.getlist('pdf')
    saved_files = []
    
    for file in files:
        if file.filename == '':
            continue
        
        if file and file.filename.endswith('.pdf'):
            filename = file.filename
            file_path = os.path.join(UPLOADS_FOLDER, filename)
            try:
                file.save(file_path)
                saved_files.append(filename)
                print(f"Successfully saved file to: {file_path}")  # Debug log
            except Exception as e:
                print(f"Error saving file: {str(e)}")  # Debug log
                return jsonify({"error": f"Error saving file {filename}: {str(e)}"}), 500
    
    if not saved_files:
        return jsonify({"error": "No valid files uploaded"}), 400
    
    return jsonify({"files": saved_files})

@api_blueprint.route("/upload/start", methods=["POST"])
def start_upload():
    data = request.json
    file_id = data['fileId']
    os.makedirs(os.path.join(TEMP_FOLDER, file_id), exist_ok=True)
    return jsonify({"status": "started"})

@api_blueprint.route("/upload/chunk", methods=["POST"])
def upload_chunk():
    file_id = request.form['fileId']
    chunk_index = request.form['chunkIndex']
    chunk = request.files['chunk']
    
    chunk_path = os.path.join(TEMP_FOLDER, file_id, f"chunk_{chunk_index}")
    chunk.save(chunk_path)
    
    return jsonify({"status": "chunk_uploaded"})

@api_blueprint.route("/upload/complete", methods=["POST"])
def complete_upload():
    file_id = request.json['fileId']
    file_name = request.json['fileName']  # Get the original filename
    temp_dir = os.path.join(TEMP_FOLDER, file_id)
    
    # Combine chunks
    chunks = sorted(os.listdir(temp_dir), key=lambda x: int(x.split('_')[1]))
    final_path = os.path.join(UPLOADS_FOLDER, file_name)  # Use original filename
    
    with open(final_path, 'wb') as outfile:
        for chunk_name in chunks:
            chunk_path = os.path.join(temp_dir, chunk_name)
            with open(chunk_path, 'rb') as infile:
                outfile.write(infile.read())
    
    # Clean up temp files
    for chunk_name in chunks:
        os.remove(os.path.join(temp_dir, chunk_name))
    os.rmdir(temp_dir)
    
    return jsonify({
        "status": "completed",
        "path": final_path,
        "fileName": file_name
    })

#dari do it ,query leave, check upload and access
@api_blueprint.route("/query", methods=["POST"])
def  query_pdf():
    data = request.json
    selected_files = data.get("selected_files", [])  # List of filenames to process
    query = data.get("query", "").strip()  # User query

    if not selected_files:
        return jsonify({"error": "No files selected."}), 400
    if not query:
        return jsonify({"error": "Query is required."}), 400

    results = []
    print(selected_files)
    for filename in selected_files:
        file_path = os.path.join(UPLOADS_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": f"File '{filename}' not found."}), 404

        # Process file with FAISS
        # index, flat_chunks, chunk_ids, model = load_pdf_to_faiss(
        #     filename, file_path, chunk_size=300, overlap=50
        # )
        # relevant_chunks = retrieve_relevant_chunks1(query,index, flat_chunks, chunk_ids)
        #response = qa_system_with_individual_summaries_faiss(index, relevant_chunks, chunk_ids, model, query)
        response=retrieve_answer(filename,query)
        results.append({
            "file": filename,
            "response": response
        })

    return jsonify({
        "message": "Query processed successfully.",
        "results": results
    })
    
    # Extract text from PDF
    #response=qa_system_with_individual_summaries_faiss(index, flat_chunks, chunk_ids, model,query)

    # Retrieve relevant chunks based on the query
    # relevant_chunks = retrieve_relevant_chunks1(query_text,index, flat_chunks, chunk_ids)
    # keywords = extract_keywords_keybert(query_text)

    # return jsonify({
    #     "query": query_text,
    #     "results": relevant_chunks,
    #     "keywords": keywords
    # })

@api_blueprint.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        session_id = data.get("session_id")
        query = data.get("query")
        file_name = data.get("file_name")
        
        if not file_name:
            return jsonify({"error": "No file name provided"}), 400
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        if not session_id:
            # Create new session
            session_id = str(uuid.uuid4())
            file_path = os.path.join(UPLOADS_FOLDER, file_name)
            
            if not os.path.exists(file_path):
                return jsonify({"error": f"File not found: {file_name}"}), 404
                
            try:
                index, flat_chunks, chunk_ids, model = load_pdf_to_faiss(
                    file_name, file_path, chunk_size=300, overlap=50
                )
                chatbot_manager.create_session(
                    session_id, file_name, index, flat_chunks, chunk_ids, model
                )
            except Exception as e:
                print(f"Error creating session: {str(e)}")
                return jsonify({"error": f"Error processing file: {str(e)}"}), 500
        
        session = chatbot_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "Invalid session ID"}), 404
        
        try:
            response = session.chat(query)
            return jsonify({
                "session_id": session_id,
                "response": response,
                "file_name": session.file_name,
                "relevant_chunks": session.get_relevant_chunks()
            })
        except Exception as e:
            print(f"Error in chat processing: {str(e)}")
            return jsonify({"error": "Error processing chat request", "details": str(e)}), 500
            
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@api_blueprint.route("/chat/history", methods=["GET"])
def get_chat_history():
    session_id = request.args.get("session_id")
    session = chatbot_manager.get_session(session_id)
    
    if not session:
        return jsonify({"error": "Invalid session ID"}), 404
    
    return jsonify({
        "session_id": session_id,
        "history": session.conversation_history,
        "file_name": session.file_name
    })

@api_blueprint.route("/chunks", methods=["GET"])
def get_chunks():
    file_name = request.args.get("file_name")
    chunk_index = request.args.get("chunk_index")
    
    if not file_name:
        return jsonify({"error": "File name is required"}), 400
        
    try:
        chunks = display_chunks(file_name)
        
        if chunk_index is not None:
            # If chunk_index is provided, return specific chunk
            chunk_idx = int(chunk_index)
            if 0 <= chunk_idx < len(chunks):
                return jsonify({
                    "chunk": {
                        "content": chunks[chunk_idx][0],
                        "keywords": chunks[chunk_idx][1],
                        "index": chunk_idx
                    }
                })
            else:
                return jsonify({"error": "Chunk index out of range"}), 404
        
        # Otherwise return all chunks
        return jsonify({
            "chunks": chunks
        })
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print(f"Error in get_chunks: {e}")
        return jsonify({"error": str(e)}), 500

@api_blueprint.route("/summarize", methods=["POST"])
def summarize_chunk():
    try:
        data = request.json
        text = data.get("text")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        summary = generate_with_flant5_individual(
            chunks=[text],
            query="Summarize this text concisely"
        )
        
        return jsonify({"summary": summary})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500

@api_blueprint.route("/embeddings", methods=["GET"])
def get_embeddings():
    file_name = request.args.get("file_name")
    if not file_name:
        return jsonify({"error": "No file name provided"}), 400
        
    try:
        # Get embeddings from FAISS index
        file_path = os.path.join(UPLOADS_FOLDER, file_name)
        index, flat_chunks, chunk_ids, model = load_pdf_to_faiss(
            file_name, file_path, chunk_size=300, overlap=50
        )
        
        # Convert embeddings to list format
        embeddings = index.reconstruct_n(0, index.ntotal)
        
        return jsonify({
            "embeddings": embeddings.tolist(),
            "chunk_ids": chunk_ids
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_blueprint.route("/knowledge-graph", methods=["GET"])
def get_knowledge_graph():
    file_name = request.args.get("file_name")
    if not file_name:
        return jsonify({"error": "File name is required"}), 400
        
    try:
        file_path = os.path.join(UPLOADS_FOLDER, file_name)
        graph = generate_knowledge_graph(file_path)
        
        if not graph:
            return jsonify({"error": "Failed to generate knowledge graph"}), 500
            
        return jsonify({"graph": graph})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
