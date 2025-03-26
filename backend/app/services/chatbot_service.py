from typing import Dict,Optional,List
import logging
from app.services.embedding_service import retrieve_relevant_chunks1,generate_comprehensive_response
logger = logging.getLogger(__name__)

class MultiDocChatSession:
    def __init__(self, files_data: Dict[str, Dict], previous_messages: Optional[List] = None, relevant_chunks: Optional[List] = None):
        """
        Initialize a chat session that handles multiple documents.
        
        Args:
            files_data: Dictionary containing processed data for each file
            previous_messages: List of previous conversation messages
            relevant_chunks: List of relevant text chunks from previous queries
        """
        self.files_data = files_data
        self.conversation_history = previous_messages or []
        self.relevant_chunks = relevant_chunks or []
        
    def chat1(self, query: str, context: Optional[Dict] = None) -> tuple:
        """Process a chat query and return response with relevant chunks."""
        try:
            all_relevant_chunks = []
            
            # Get relevant chunks from all documents
            for file_name, data in self.files_data.items():
                chunks = retrieve_relevant_chunks1(
                    query=query,
                    index=data['index'],
                    flat_chunks=data['flat_chunks'],
                    chunk_ids=data['chunk_ids'],
                    model=data['model']
                )
                
                # Add source file information to chunks
                for chunk in chunks:
                    chunk['source_file'] = file_name
                all_relevant_chunks.extend(chunks)
            
            # Sort chunks by relevance
            all_relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Generate comprehensive response
            response = generate_comprehensive_response(
                chunks=all_relevant_chunks,
                query=query,
                #conversation_history=self.conversation_history
            )
            
            # Update conversation history
            self.conversation_history.extend([
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': response}
            ])
            
            # Update relevant chunks
            self.relevant_chunks = all_relevant_chunks
            
            return response, all_relevant_chunks
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}", exc_info=True)
            raise

    def get_file_names(self) -> List[str]:
        """Get list of file names in the session."""
        return list(self.files_data.keys())
        
    def get_relevant_chunks(self) -> List[Dict]:
        """Get relevant chunks from the last query."""
        return self.relevant_chunks

class ChatbotManager:
    def __init__(self):
        """Initialize the ChatbotManager with an empty sessions dictionary."""
        self.sessions: Dict[str, MultiDocChatSession] = {}
        
    
    def create_session(self, session_id: str, files_data: Dict[str, Dict],
                      previous_messages: Optional[List] = None,
                      relevant_chunks: Optional[List] = None) -> None:
        """
        Create a new chat session with the given data.
        This is the main method to create sessions.
        """
        try:
            self.sessions[session_id] = MultiDocChatSession(
                files_data=files_data,
                previous_messages=previous_messages,
                relevant_chunks=relevant_chunks
            )
            logger.info(f"Created new session with ID: {session_id}")
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}", exc_info=True)
            raise

    def get_session(self, session_id: str) -> Optional[MultiDocChatSession]:
        """Get a chat session by its ID."""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session by its ID."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions
# from typing import List, Dict, Optional
# import json
# from app.services.embedding_service import retrieve_relevant_chunks1
# from app.services.summarization import generate_with_flant5_individual

# class ChatSession:
#     def __init__(self, file_name: str, index, flat_chunks, chunk_ids, model):
#         self.file_name = file_name
#         self.index = index
#         self.flat_chunks = flat_chunks
#         self.chunk_ids = chunk_ids
#         self.model = model
#         self.conversation_history = []
#         self.last_relevant_chunks = []
        
#     def chat(self, query: str) -> str:
#         try:
#             # Get relevant chunks
#             relevant_chunk_indices = retrieve_relevant_chunks1(
#                 query,
#                 self.index,
#             )
            
#             # Get the actual chunks - Convert indices to integers and handle them directly
#             relevant_chunks = []
#             for idx in relevant_chunk_indices:
#                 if 0 <= idx < len(self.flat_chunks):  # Check if index is valid
#                     chunk = self.flat_chunks[idx]
#                     relevant_chunks.append(chunk)
            
#             if not relevant_chunks:
#                 return "I couldn't find any relevant information in the document."
            
#             # Store for later retrieval
#             self.last_relevant_chunks = [
#                 {"content": chunk, "summary": None} 
#                 for chunk in relevant_chunks
#             ]

#             # Generate response using the chunks
#             response = generate_with_flant5_individual(
#                 chunks=relevant_chunks,
#                 query=query
#             )

#             # Store in conversation history
#             self.conversation_history.append({
#                 "query": query,
#                 "response": response,
#                 "chunks": relevant_chunks
#             })

#             return response

#         except Exception as e:
#             print(f"Error in chat: {str(e)}")
#             print(f"Debug info - chunk_ids: {len(self.chunk_ids)}, flat_chunks: {len(self.flat_chunks)}")
#             print(f"Retrieved indices: {relevant_chunk_indices}")
#             raise
    
#     def get_relevant_chunks(self):
#         return self.last_relevant_chunks

# class ChatbotManager:
#     def __init__(self):
#         self.active_sessions = {}
    
#     def create_session(self, session_id: str, file_name: str, index, flat_chunks, chunk_ids, model):
#         """Create a new chat session."""
#         self.active_sessions[session_id] = ChatSession(
#             file_name=file_name,
#             index=index,
#             flat_chunks=flat_chunks,
#             chunk_ids=chunk_ids,
#             model=model
#         )
    
#     def get_session(self, session_id: str) -> Optional[ChatSession]:
#         """Retrieve an existing chat session."""
#         return self.active_sessions.get(session_id)
    
#     def remove_session(self, session_id: str):
#         """Remove a chat session."""
#         if session_id in self.active_sessions:
#             del self.active_sessions[session_id] 