import React, { useState } from 'react';
import ChatInterface from './ChatInterface';
import DocumentSelection from './DocumentSelection';
import DocumentView from './DocumentView';
import '../styles/MainContent.css';

function MainContent() {
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [activeTab, setActiveTab] = useState('chat');
  const [selectedDocument, setSelectedDocument] = useState(null);

  const handleDocumentSelect = (document) => {
    setSelectedDocument(document);
    setActiveTab(`doc_${document.id}`);
  };

  return (
    <div className="main-content">
      {selectedDocs.length === 0 ? (
        <DocumentSelection onDocsSelected={setSelectedDocs} />
      ) : (
        <>
          <div className="tabs-header">
            <button 
              className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              Chat with document
            </button>
            {selectedDocs.map((doc) => (
              <button
                key={doc.id}
                className={`tab ${activeTab === `doc_${doc.id}` ? 'active' : ''}`}
                onClick={() => handleDocumentSelect(doc)}
              >
                {doc.name}
              </button>
            ))}
          </div>

          <div className="tab-content">
            {activeTab === 'chat' ? (
              <ChatInterface 
                selectedDocs={selectedDocs}
                onChunkSelect={(chunk) => console.log('Selected chunk:', chunk)}
              />
            ) : (
              <DocumentView 
                document={selectedDocument}
                onChunkSelect={(chunk) => console.log('Selected chunk:', chunk)}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default MainContent; 