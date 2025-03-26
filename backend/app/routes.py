# API route definitions
# routes.py
import os
from typing import Dict,Optional,List
from flask import Blueprint, request, jsonify ,send_file

from app.services.pdf_processing import extract_text_from_pdf
from app.services.chunking import agentic_chunking
from app.services.embedding_service import (load_pdf_to_faiss,retrieve_comprehensive_answer,generate_comprehensive_response, 
                                            retrieve_relevant_chunks1)
from app.services.keyword_extraction import extract_keywords_keybert
from app.services.query_handling import  retrieve_answer  ,qa_system_with_individual_summaries_faiss, display_chunks
from app.services.chatbot_service import ChatbotManager
from app.services.summarization import generate_with_flant5_individual
import uuid
from app.services.knowledge_graph import generate_enhanced_knowledge_graph
from werkzeug.utils import secure_filename
import logging 
import fitz # PyMuPDF
import json
api_blueprint = Blueprint("api", __name__)
logger = logging.getLogger(__name__)
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
@api_blueprint.route("/query", methods=["POST"])
def query_pdf():
    data = request.json
    selected_files = data.get("selected_files", [])
    query = data.get("query", "").strip()
    if not selected_files:
        return jsonify({"error": "No files selected."}), 400
    if not query:
        return jsonify({"error": "Query is required."}), 400

    try:
        # Now we'll get one comprehensive answer instead of per-file results
        comprehensive_response = retrieve_comprehensive_answer(selected_files, query)
        
        return jsonify({
            "message": "Query processed successfully.",
            "response": comprehensive_response
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@api_blueprint.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        session_id = data.get("session_id")
        query = data.get("query")
        selected_files = data.get("selected_files", [])
        context = data.get("context", {})
        
        # Validate inputs
        if not selected_files:
            return jsonify({"error": "No files selected"}), 400
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Create or get session
        if not session_id:
            session_id = str(uuid.uuid4())
            # Initialize session with multiple documents
            try:
                session_data = {}
                for file_name in selected_files:
                    file_path = os.path.join(UPLOADS_FOLDER, file_name)
                    
                    if not os.path.exists(file_path):
                        return jsonify({"error": f"File not found: {file_name}"}), 404
                    
                    # Load and process each document
                    index, flat_chunks, chunk_ids, model = load_pdf_to_faiss(
                        file_name,
                        file_path,
                        chunk_size=300,
                        overlap=50
                    )
                    
                    session_data[file_name] = {
                        'index': index,
                        'flat_chunks': flat_chunks,
                        'chunk_ids': chunk_ids,
                        'model': model
                    }
                
                # Create session with multiple documents
                chatbot_manager.create_session(
                    session_id=session_id,
                    files_data=session_data,
                    previous_messages=context.get('previous_messages', []),
                    relevant_chunks=context.get('relevant_chunks', [])
                )
                
            except Exception as e:
                logger.error(f"Error creating session: {str(e)}", exc_info=True)
                return jsonify({
                    "error": "Error processing files",
                    "details": str(e)
                }), 500
        
        # Get existing session
        session = chatbot_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "Invalid session ID"}), 404
        
        try:
            # Process chat with context
            response, relevant_chunks = session.chat1(
                query=query,
                context=context
            )
            
            return jsonify({
                "session_id": session_id,
                "response": response,
                "relevant_chunks": relevant_chunks,
                "files": selected_files
            })
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}", exc_info=True)
            return jsonify({
                "error": "Error processing chat request",
                "details": str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Server error",
            "details": str(e)
        }), 500

@api_blueprint.route("/chat/history", methods=["GET"])
def get_chat_history():
    try:
        session_id = request.args.get("session_id")
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
            
        session = chatbot_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "Invalid session ID"}), 404
            
        return jsonify({
            "session_id": session_id,
            "history": session.conversation_history,
            "files": session.get_file_names(),
            "relevant_chunks": session.get_relevant_chunks()
        })
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Error retrieving chat history",
            "details": str(e)
        }), 500
@api_blueprint.route("/pdf-text", methods=["GET"])
def get_pdf_text():
    file_name = request.args.get('file_name')
    if not file_name:
        return jsonify({"error": "No file name provided"}), 400
    print(UPLOADS_FOLDER,file_name)
    try:
        file_path = os.path.join(UPLOADS_FOLDER, file_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Extract text from PDF
        doc = fitz.open(file_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pages.append(page.get_text())
        
        return jsonify({"pages": pages, "page_count": len(pages)})
    
    except Exception as e:
        logger.error(f"Error accessing PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_blueprint.route("/highlight-pdf", methods=["POST"])
def highlight_pdf():
    data = request.json
    file_name = data.get('fileName')
    highlights = data.get('highlights', [])
    node_types = data.get('nodeTypes', [])
    graph_data = data.get('graphData', {})
    
    if not file_name:
        return jsonify({"error": "No file name provided"}), 400
    
    try:
        file_path = os.path.join(UPLOADS_FOLDER, file_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Define colors for different node types (R, G, B)
        colors = {
            "DOCUMENT": (1, 1, 1, 0.3),  # White (transparent)
            "MAIN_TOPIC": (0.3, 0.7, 0.3, 0.3),  # Green
            "SUBTOPIC": (0.13, 0.59, 0.95, 0.3),  # Blue
            "CONCEPT": (1, 0.76, 0.03, 0.3),  # Yellow
            "ENTITY": (0.91, 0.12, 0.39, 0.3)  # Pink
        }
        
        # Open the PDF
        doc = fitz.open(file_path)
        
        # Dictionary to store processed highlights to avoid duplicates
        processed = {}
        
        # Apply highlights
        for highlight in highlights:
            page_num = highlight.get('page')
            text = highlight.get('text')
            highlight_type = highlight.get('type')
            
            # Skip if invalid data or node type not selected
            if not all([page_num, text, highlight_type]) or highlight_type not in node_types:
                continue
            
            # Adjust for 0-based indexing
            page_num = int(page_num) - 1
            if page_num < 0 or page_num >= len(doc):
                continue
            
            # Skip if already processed this text on this page
            key = f"{page_num}_{text}"
            if key in processed:
                continue
            
            processed[key] = True
            
            page = doc[page_num]
            
            # Search for text instances on the page
            text_instances = page.search_for(text)
            
            # Apply highlight to all instances
            for rect in text_instances:
                # Use standard highlighting method
                color = colors.get(highlight_type, (1, 1, 0, 0.5))  # Default to yellow
                
                # Create a highlight annotation with proper settings
                annot = page.add_highlight_annot(rect)
                
                # Set color based on node type
                annot.set_colors(stroke=color)
                
                # Add metadata to make it recognizable by other tools
                info = {
                    "title": f"Highlight: {text}",
                    "subject": highlight_type,
                    "content": text
                }
                annot.set_info(**info)
                
                # Set appearance to ensure it's visible in all viewers
                annot.update(opacity=color[3])
        
        # Create output path for the highlighted PDF
        output_dir = os.path.join(UPLOADS_FOLDER, 'highlighted')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"highlighted_{file_name}")
        
        # Add graph metadata as an appendix
        def create_graph_metadata_page(doc, graph_data):
            """Create multiple pages with graph metadata to prevent content cutting"""
            # Define types in order of importance
            type_order = ["MAIN_TOPIC", "SUBTOPIC", "CONTENT_TYPES", "CONCEPT", "ENTITY"]
            
            # Fonts
            title_font = {"fontsize": 16, "fontname": "helv", "color": (0,0,0)}
            section_font = {"fontsize": 14, "fontname": "helv", "color": (0,0.5,0.5)}
            node_font = {"fontsize": 12, "fontname": "helv", "color": (0,0,0)}
            terms_font = {"fontsize": 10, "fontname": "helv", "color": (0.3,0.3,0.3)}
            
            def wrap_text(text, max_width=500):
                """Wrap long text to prevent overflow"""
                wrapped_lines = []
                words = text.split()
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) > max_width / 8:  # Rough character width estimation
                        wrapped_lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1
                
                if current_line:
                    wrapped_lines.append(' '.join(current_line))
                
                return wrapped_lines
            
            # First page with title
            page = doc.new_page()
            page.insert_text((50, 50), "Knowledge Graph Metadata", **title_font)
            
            y_position = 100
            page_margin = 50
            page_width = page.rect.width
            page_height = page.rect.height
            line_height = 25
            
            for node_type in type_order:
                # Filter nodes of this type
                type_nodes = [node for node in graph_data.get('nodes', []) if node.get('type') == node_type]
                
                if not type_nodes:
                    continue
                
                # Check if we need a new page
                if y_position > page_height - 150:
                    page = doc.new_page()
                    y_position = page_margin
                
                # Add type header
                page.insert_text(
                    (page_margin, y_position), 
                    f"{node_type.replace('_', ' ').title()} Details", 
                    **section_font
                )
                y_position += line_height * 1.5
                
                for node in type_nodes:
                    # Node label with wrapping
                    label = node.get('label', 'Unnamed')
                    label_lines = wrap_text(label)
                    
                    for line in label_lines:
                        # Check if we need a new page
                        if y_position > page_height - 100:
                            page = doc.new_page()
                            y_position = page_margin
                        
                        page.insert_text(
                            (page_margin + 20, y_position), 
                            f"• {line}", 
                            **node_font
                        )
                        y_position += line_height
                    
                    # Related terms with wrapping
                    terms = node.get('terms', [])
                    if terms:
                        # Check if we need a new page
                        if y_position > page_height - 100:
                            page = doc.new_page()
                            y_position = page_margin
                        
                        terms_text = "Related Terms: " + ", ".join(terms)
                        terms_lines = wrap_text(terms_text)
                        
                        for line in terms_lines:
                            # Check if we need a new page
                            if y_position > page_height - 100:
                                page = doc.new_page()
                                y_position = page_margin
                            
                            page.insert_text(
                                (page_margin + 40, y_position), 
                                line, 
                                **terms_font
                            )
                            y_position += line_height
                    
                    # Add some extra space between nodes
                    y_position += line_height
        
        # Add graph metadata page
        create_graph_metadata_page(doc, graph_data)
        
        # Save the highlighted PDF with proper options
        doc.save(
            output_path,
            garbage=4,         # Clean up redundant objects
            deflate=True,      # Compress streams
            pretty=False,      # No pretty printing (smaller file)
            clean=True,        # Clean up content
            linear=True        # Optimize for web viewing
        )
        doc.close()
        
        # Return the highlighted PDF as a download
        return send_file(output_path, as_attachment=True, download_name=f"highlighted_{file_name}")
    
    except Exception as e:
        logger.error(f"Error highlighting PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500
@api_blueprint.route("/pdf", methods=["GET"])
def serve_pdf():
    file_name = request.args.get('file_name')
    if not file_name:
        return jsonify({"error": "No file name provided"}), 400
    
    try:
        file_path = os.path.join(UPLOADS_FOLDER, file_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Return the PDF file directly
        return send_file(file_path, mimetype='application/pdf')
    
    except Exception as e:
        logger.error(f"Error serving PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
            "embeddings": embeddings.tolist(),
            "chunk_ids": chunk_ids
        })
   
@api_blueprint.route("/knowledge-graph", methods=["GET"])
def get_knowledge_graph():
    file_name = request.args.get("file_name")
    if not file_name:
        return jsonify({"error": "File name is required"}), 400
        
    try:
        file_path = os.path.join(UPLOADS_FOLDER, file_name)
        graph = generate_enhanced_knowledge_graph(file_path)
        
        if not graph:
            return jsonify({"error": "Failed to generate knowledge graph"}), 500
        
        # Convert NumPy float32 values to Python floats
        graph = convert_numpy_to_python_types(graph)
        
        return jsonify({"graph": graph})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def convert_numpy_to_python_types(obj):
    """Convert NumPy types to Python native types to make them JSON serializable."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_to_python_types(obj.tolist())
    else:
        return obj
#dari do it ,query leave, check upload and access
# @api_blueprint.route("/query", methods=["POST"])
# def  query_pdf():
#     data = request.json
#     selected_files = data.get("selected_files", [])  # List of filenames to process
#     query = data.get("query", "").strip()  # User query

#     if not selected_files:
#         return jsonify({"error": "No files selected."}), 400
#     if not query:
#         return jsonify({"error": "Query is required."}), 400

#     results = []
#     print(selected_files)
#     for filename in selected_files:
#         file_path = os.path.join(UPLOADS_FOLDER, filename)
#         if not os.path.exists(file_path):
#             return jsonify({"error": f"File '{filename}' not found."}), 404

#         # Process file with FAISS
#         # index, flat_chunks, chunk_ids, model = load_pdf_to_faiss(
#         #     filename, file_path, chunk_size=300, overlap=50
#         # )
#         # relevant_chunks = retrieve_relevant_chunks1(query,index, flat_chunks, chunk_ids)
#         #response = qa_system_with_individual_summaries_faiss(index, relevant_chunks, chunk_ids, model, query)
#         response=retrieve_answer(filename,query)
#         results.append({
#             "file": filename,
#             "response": response
#         })

#     return jsonify({
#         "message": "Query processed successfully.",
#         "results": results
#     })
    
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
# Helper class for managing chat sessions with multiple documents


# Initialize the global ChatbotManager instance
# @api_blueprint.route("/chat", methods=["POST"])
# def chat():
#     try:
#         data = request.json
#         session_id = data.get("session_id")
#         query = data.get("query")
#         file_name = data.get("file_name")
        
#         if not file_name:
#             return jsonify({"error": "No file name provided"}), 400
        
#         if not query:
#             return jsonify({"error": "No query provided"}), 400
        
#         if not session_id:
#             # Create new session
#             session_id = str(uuid.uuid4())
#             file_path = os.path.join(UPLOADS_FOLDER, file_name)
            
#             if not os.path.exists(file_path):
#                 return jsonify({"error": f"File not found: {file_name}"}), 404
                
#             try:
#                 index, flat_chunks, chunk_ids, model = load_pdf_to_faiss(
#                     file_name, file_path, chunk_size=300, overlap=50
#                 )
#                 chatbot_manager.create_session(
#                     session_id, file_name, index, flat_chunks, chunk_ids, model
#                 )
#             except Exception as e:
#                 print(f"Error creating session: {str(e)}")
#                 return jsonify({"error": f"Error processing file: {str(e)}"}), 500
        
#         session = chatbot_manager.get_session(session_id)
#         if not session:
#             return jsonify({"error": "Invalid session ID"}), 404
        
#         try:
#             response = session.chat(query)
#             return jsonify({
#                 "session_id": session_id,
#                 "response": response,
#                 "file_name": session.file_name,
#                 "relevant_chunks": session.get_relevant_chunks()
#             })
#         except Exception as e:
#             print(f"Error in chat processing: {str(e)}")
#             return jsonify({"error": "Error processing chat request", "details": str(e)}), 500
            
#     except Exception as e:
#         print(f"Error in chat endpoint: {str(e)}")
#         return jsonify({"error": f"Server error: {str(e)}"}), 500

# @api_blueprint.route("/chat/history", methods=["GET"])
# def get_chat_history():
#     session_id = request.args.get("session_id")
#     session = chatbot_manager.get_session(session_id)
    
#     if not session:
#         return jsonify({"error": "Invalid session ID"}), 404
    
#     return jsonify({
#         "session_id": session_id,
#         "history": session.conversation_history,
#         "file_name": session.file_name
#     })
