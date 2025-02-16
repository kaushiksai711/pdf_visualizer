import React, { useState, useRef, useEffect } from 'react';
import '../styles/ChatPanel.css';

function ChatPanel({ instance, updateInstanceData, onRemoveDoc }) {
  const [input, setInput] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [instance?.messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || !instance || !instance.documents.length) return;

    const newMessage = { type: 'user', content: input };
    const updatedMessages = [...(instance.messages || []), newMessage];
    
    updateInstanceData(
      instance.sessionId, 
      updatedMessages,
      instance.relevantChunks
    );
    
    setInput('');

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: instance.sessionId,
          query: input,
          file_name: instance.documents[0].name // Using first document for now
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (!instance.sessionId && data.session_id) {
        updateInstanceData(
          data.session_id,
          [...updatedMessages, { type: 'bot', content: data.response }],
          data.relevant_chunks || []
        );
      } else {
        updateInstanceData(
          instance.sessionId,
          [...updatedMessages, { type: 'bot', content: data.response }],
          data.relevant_chunks || []
        );
      }
    } catch (error) {
      console.error('Error:', error);
      updateInstanceData(
        instance.sessionId,
        [...updatedMessages, { 
          type: 'bot', 
          content: 'Error: Failed to get response from server.' 
        }],
        instance.relevantChunks
      );
    }
  };

  const handleSummarize = async (chunk) => {
    try {
      const response = await fetch('http://localhost:5000/api/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: chunk.content })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const updatedChunks = instance.relevantChunks.map(c => 
        c === chunk ? { ...c, summary: data.summary } : c
      );
      
      updateInstanceData(instance.sessionId, instance.messages, updatedChunks);
    } catch (error) {
      console.error('Error summarizing chunk:', error);
    }
  };

  if (!instance) {
    return (
      <div className="panel chat-panel">
        <div className="panel-header">
          <h2>Chat with Documents</h2>
        </div>
        <div className="panel-content">
          <div className="no-instance-message">
            Please select or create a new instance to start chatting
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="panel chat-panel" style={{ background: 'rgba(0, 0, 0, 0.7)', borderRadius: '10px', boxShadow: '0 0 20px rgba(255, 255, 255, 0.5)' }}>
      <div className="panel-header">
        <h2 style={{ color: '#ffffff' }}>Chat with Documents - {instance.name}</h2>
        <div className="selected-docs">
          {instance.documents.map(doc => (
            <div key={doc.id} className="doc-tag">
              {doc.name}
              <button onClick={() => onRemoveDoc(doc.id)}>&times;</button>
            </div>
          ))}
        </div>
      </div>
      
      <div className="panel-content">
        <div className="chat-messages">
          {instance.messages?.map((msg, idx) => (
            <div key={idx} className={`message ${msg.type}`}>
              {msg.content}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <div className="relevant-chunks">
          {instance.relevantChunks?.map((chunk, idx) => (
            <div key={idx} className="chunk">
              <div className="chunk-content">{chunk.content}</div>
              {chunk.summary ? (
                <div className="chunk-summary">{chunk.summary}</div>
              ) : (
                <button 
                  className="summarize-btn"
                  onClick={() => handleSummarize(chunk)}
                >
                  Summarize
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder={instance.documents.length ? "Ask a question..." : "Please upload documents first"}
          disabled={!instance.documents.length}
        />
        <button 
          onClick={handleSendMessage}
          disabled={!instance.documents.length}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatPanel; 