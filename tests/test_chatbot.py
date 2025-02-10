import requests
import os

# Test configuration
BASE_URL = "http://localhost:5000"  # Adjust if your server runs on a different port
TEST_PDF_PATH = "D:\\vis_pdf\\pdf_query_system\\backend\\app\\uploads\\Assignment for Internship-AI ML.pdf"  # Replace with an actual PDF path

def test_chatbot_workflow():
    """
    Test the complete chatbot workflow:
    1. Upload PDF
    2. Start chat session
    3. Send multiple queries
    4. Check conversation history
    """
    
    # Step 1: Upload PDF
    with open(TEST_PDF_PATH, 'rb') as pdf_file:
        files = {'pdf': pdf_file}
        upload_response = requests.post(
            f"{BASE_URL}/upload",
            files=files
        )
    
    assert upload_response.status_code == 200
    print("[PASS] PDF Upload successful")
    
    # Get the filename from the upload response
    file_name = os.path.basename(TEST_PDF_PATH)
    
    # Step 2: Start chat session (first query)
    initial_query = "What is the main topic of this document?"
    chat_response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "session_id": None,  # First query, no session yet
            "query": initial_query,
            "file_name": file_name
        }
    )
    
    assert chat_response.status_code == 200
    chat_data = chat_response.json()
    session_id = chat_data["session_id"]
    print("[PASS] Chat session created")
    print(f"Initial response: {chat_data['response']}\n")
    
    # Step 3: Follow-up queries using the same session
    follow_up_queries = [
        "Can you provide more details about that?",
        "What are the key findings or conclusions?",
        "How does this relate to your previous answers?"
    ]
    
    for query in follow_up_queries:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "session_id": session_id,
                "query": query,
                "file_name": file_name
            }
        )
        
        assert response.status_code == 200
        print(f"Query: {query}")
        print(f"Response: {response.json()['response']}\n")
    
    # Step 4: Check conversation history
    history_response = requests.get(
        f"{BASE_URL}/chat/history",
        params={"session_id": session_id}
    )
    
    assert history_response.status_code == 200
    history_data = history_response.json()
    
    print("Conversation History:")
    for exchange in history_data["history"]:
        role = exchange["role"]
        content = exchange["content"]
        print(f"{role.upper()}: {content}\n")
        
        if role == "assistant":
            print("Thought Process:")
            for thought in exchange["thought_process"]:
                print(f"  {thought}")
            print(f"Relevant Chunks: {exchange['relevant_chunks']}\n")

if __name__ == "__main__":
    try:
        test_chatbot_workflow()
        print("All tests completed successfully!")
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"Error during testing: {str(e)}") 