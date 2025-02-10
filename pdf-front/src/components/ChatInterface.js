import React, { useState, useRef, useEffect } from 'react';
import '../styles/ChatInterface.css';

function ChatInterface({ selectedDocs }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [relevantChunks, setRelevantChunks] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const chatEndRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessage = {
      type: 'user',
      content: input
    };

    setMessages([...messages, newMessage]);
    setInput('');

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          query: input,
          file_name: selectedDocs[0]?.name // Currently using first document
        }),
      });

      const data = await response.json();
      
      if (!sessionId && data.session_id) {
        setSessionId(data.session_id);
      }

      setMessages(prev => [...prev, {
        type: 'bot',
        content: data.response
      }]);
      
      setRelevantChunks(data.relevant_chunks || []);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.type}`}>
            {message.content}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      <div className="relevant-chunks">
        <h3>Relevant Chunks</h3>
        {relevantChunks.map((chunk, index) => (
          <div key={index} className="chunk">
            <div className="chunk-content">{chunk.content}</div>
            {chunk.summary && (
              <div className="chunk-summary">{chunk.summary}</div>
            )}
          </div>
        ))}
      </div>

      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default ChatInterface; 