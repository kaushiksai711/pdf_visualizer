import React, { useState, useEffect, useCallback } from 'react';
import '../styles/DocumentView.css';

function DocumentView({ document, onChunkSelect }) {
  const [chunks, setChunks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadDocumentChunks = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:5000/api/chunks?file_name=${document.name}`);
      if (!response.ok) {
        throw new Error('Failed to load chunks');
      }
      const data = await response.json();
      setChunks(data.chunks || []);
    } catch (error) {
      console.error('Error loading chunks:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  }, [document?.name]);

  useEffect(() => {
    if (document) {
      loadDocumentChunks();
    }
  }, [document, loadDocumentChunks]);

  if (loading) {
    return <div className="loading">Loading document chunks...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  return (
    <div className="document-view">
      <div className="document-info">
        <h2>{document.name}</h2>
        <div className="document-metadata">
          <p>Path: {document.path}</p>
          <p>Total Chunks: {chunks.length}</p>
        </div>
      </div>

      <div className="chunks-container">
        <h3>Document Chunks</h3>
        {chunks.map((chunk, index) => (
          <div 
            key={index}
            className="chunk-item"
            onClick={() => onChunkSelect(chunk)}
          >
            <div className="chunk-content">{chunk[0]}</div>
            <div className="chunk-keywords">
              <strong>Keywords:</strong> {Array.isArray(chunk[1]) ? chunk[1].join(', ') : chunk[1]}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default DocumentView; 