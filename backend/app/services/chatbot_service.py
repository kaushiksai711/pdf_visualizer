from typing import List, Dict, Optional
import json
from app.services.embedding_service import retrieve_relevant_chunks1
from app.services.summarization import generate_with_flant5_individual

class ChatSession:
    def __init__(self, file_name: str, index, flat_chunks, chunk_ids, model):
        self.file_name = file_name
        self.index = index
        self.flat_chunks = flat_chunks
        self.chunk_ids = chunk_ids
        self.model = model
        self.conversation_history = []
        self.last_relevant_chunks = []
        
    def chat(self, query: str) -> str:
        try:
            # Get relevant chunks
            relevant_chunk_indices = retrieve_relevant_chunks1(
                query,
                self.index,
            )
            
            # Get the actual chunks - Convert indices to integers and handle them directly
            relevant_chunks = []
            for idx in relevant_chunk_indices:
                if 0 <= idx < len(self.flat_chunks):  # Check if index is valid
                    chunk = self.flat_chunks[idx]
                    relevant_chunks.append(chunk)
            
            if not relevant_chunks:
                return "I couldn't find any relevant information in the document."
            
            # Store for later retrieval
            self.last_relevant_chunks = [
                {"content": chunk, "summary": None} 
                for chunk in relevant_chunks
            ]

            # Generate response using the chunks
            response = generate_with_flant5_individual(
                chunks=relevant_chunks,
                query=query
            )

            # Store in conversation history
            self.conversation_history.append({
                "query": query,
                "response": response,
                "chunks": relevant_chunks
            })

            return response

        except Exception as e:
            print(f"Error in chat: {str(e)}")
            print(f"Debug info - chunk_ids: {len(self.chunk_ids)}, flat_chunks: {len(self.flat_chunks)}")
            print(f"Retrieved indices: {relevant_chunk_indices}")
            raise
    
    def get_relevant_chunks(self):
        return self.last_relevant_chunks

class ChatbotManager:
    def __init__(self):
        self.active_sessions = {}
    
    def create_session(self, session_id: str, file_name: str, index, flat_chunks, chunk_ids, model):
        """Create a new chat session."""
        self.active_sessions[session_id] = ChatSession(
            file_name=file_name,
            index=index,
            flat_chunks=flat_chunks,
            chunk_ids=chunk_ids,
            model=model
        )
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieve an existing chat session."""
        return self.active_sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Remove a chat session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id] 