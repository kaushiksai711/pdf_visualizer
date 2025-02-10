import React, { useState } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import DocumentPanel from './components/DocumentPanel';
import ChatPanel from './components/ChatPanel';
import RagVisualizer from './components/RagVisualizer';
import './styles/App.css';

function App() {
  const [activeInstance, setActiveInstance] = useState(null);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [visualizerExpanded, setVisualizerExpanded] = useState(false);

  const updateInstanceData = (sessionId, messages, relevantChunks) => {
    setActiveInstance(prev => {
      if (!prev) return null;
      return {
        ...prev,
        sessionId,
        messages,
        relevantChunks
      };
    });
  };

  return (
    <Router>
      <div className={`app-container ${visualizerExpanded ? 'expanded-visualizer' : ''}`}>
        <DocumentPanel 
          activeInstance={activeInstance}
          setActiveInstance={setActiveInstance}
          selectedDocs={selectedDocs}
          setSelectedDocs={setSelectedDocs}
        />
        <ChatPanel 
          instance={activeInstance}
          updateInstanceData={updateInstanceData}
          onRemoveDoc={(docId) => {
            setSelectedDocs(docs => docs.filter(d => d.id !== docId));
          }}
        />
        <RagVisualizer 
          isExpanded={visualizerExpanded}
          onToggleExpand={() => setVisualizerExpanded(!visualizerExpanded)}
          selectedDocs={selectedDocs}
        />
      </div>
    </Router>
  );
}

export default App; 