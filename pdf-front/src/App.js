"use client"

import { useState } from "react"
import { BrowserRouter as Router, Route, Routes } from "react-router-dom"
import DocumentPanel from "./components/DocumentPanel"
import ChatPanel from "./components/ChatPanel"
import RagVisualizer from "./components/RagVisualizer"
import Yggdrasil from "./components/Yggdrasil.tsx"
import KnowledgeGraph from "./components/KnowledgeGraph"
import "./styles/App.css"
import "./styles/Yggdrasil.css"

function App() {
  const [activeInstance, setActiveInstance] = useState(null)
  const [selectedDocs, setSelectedDocs] = useState([])
  const [visualizerExpanded, setVisualizerExpanded] = useState(false)

  const updateInstanceData = (sessionId, messages, relevantChunks) => {
    setActiveInstance((prev) => {
      if (!prev) return null
      return {
        ...prev,
        sessionId,
        messages,
        relevantChunks,
      }
    })
  }

  return (
    <Router>
      <div className={`app-container ${visualizerExpanded ? "expanded-visualizer" : ""}`}>
        <Routes>
          <Route path="/knowledge-graph/:fileName" element={<KnowledgeGraph/>} />
          <Route path="/" element={
            <>
              <Yggdrasil className="fixed inset-0 -z-10" />
              <div className="content-wrapper">
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
                    setSelectedDocs((docs) => docs.filter((d) => d.id !== docId))
                  }}
                />
                <RagVisualizer
                  isExpanded={visualizerExpanded}
                  onToggleExpand={() => setVisualizerExpanded(!visualizerExpanded)}
                  selectedDocs={selectedDocs}
                />
              </div>
            </>
          } />
        </Routes>
      </div>
    </Router>
  )
}

export default App


// import React, { useState ,useEffect} from 'react';
// import { BrowserRouter as Router } from 'react-router-dom';
// import DocumentPanel from './components/DocumentPanel';
// import ChatPanel from './components/ChatPanel';
// import RagVisualizer from './components/RagVisualizer';
// import Yggdrasil from './components/Yggdrasil';
// import './styles/App.css';


// function App() {
//   const [activeInstance, setActiveInstance] = useState(null);
//   const [selectedDocs, setSelectedDocs] = useState([]);
//   const [visualizerExpanded, setVisualizerExpanded] = useState(false);
//   useEffect(() => {
//     const starsContainer = document.querySelector('.stars');
//     if (!starsContainer) return;
  
//     const numStars = 150;
//     const starElements = [];
  
//     // Create Stars
//     for (let i = 0; i < numStars; i++) {
//       const star = document.createElement('div');
//       star.className = 'star';
//       const size = Math.random() * 2 + 1;
//       star.style.width = `${size}px`;
//       star.style.height = `${size}px`;
//       star.style.top = `${Math.random() * 100}vh`;
//       star.style.left = `${Math.random() * 100}vw`;
//       starsContainer.appendChild(star);
//       starElements.push(star);
//     }
  
//     // Create Constellations by connecting some stars
//     const numConnections = 20;
//     for (let i = 0; i < numConnections; i++) {
//       const star1 = starElements[Math.floor(Math.random() * numStars)];
//       const star2 = starElements[Math.floor(Math.random() * numStars)];
  
//       if (star1 && star2 && star1 !== star2) {
//         const line = document.createElement('div');
//         line.className = 'constellation';
  
//         const x1 = star1.offsetLeft;
//         const y1 = star1.offsetTop;
//         const x2 = star2.offsetLeft;
//         const y2 = star2.offsetTop;
//         const length = Math.hypot(x2 - x1, y2 - y1);
//         const angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);
  
//         line.style.width = `${length}px`;
//         line.style.transform = `rotate(${angle}deg)`;
//         line.style.left = `${x1}px`;
//         line.style.top = `${y1}px`;
  
//         starsContainer.appendChild(line);
//       }
//     }
  
//     // Add Shooting Stars
//     setInterval(() => {
//       const shootingStar = document.createElement('div');
//       shootingStar.className = 'shooting-star';
//       shootingStar.style.top = `${Math.random() * 100}vh`;
//       shootingStar.style.left = `${Math.random() * 100}vw`;
//       starsContainer.appendChild(shootingStar);
      
//       setTimeout(() => shootingStar.remove(), 2000);
//     }, 3000);
//   }, []);
  
  
//   const updateInstanceData = (sessionId, messages, relevantChunks) => {
//     setActiveInstance(prev => {
//       if (!prev) return null;
//       return {
//         ...prev,
//         sessionId,
//         messages,
//         relevantChunks
//       };
//     });
//   };

//   return (
//     <Router>
//       <div className={`app-container ${visualizerExpanded ? 'expanded-visualizer' : ''}`}>
//       <div className="stars"></div> {/* Ensure this exists */}
//       <Yggdrasil/ >
//         <DocumentPanel 
//           activeInstance={activeInstance}
//           setActiveInstance={setActiveInstance}
//           selectedDocs={selectedDocs}
//           setSelectedDocs={setSelectedDocs}
//         />
//         <ChatPanel 
//           instance={activeInstance}
//           updateInstanceData={updateInstanceData}
//           onRemoveDoc={(docId) => {
//             setSelectedDocs(docs => docs.filter(d => d.id !== docId));
//           }}
//         />
//         <RagVisualizer 
//           isExpanded={visualizerExpanded}
//           onToggleExpand={() => setVisualizerExpanded(!visualizerExpanded)}
//           selectedDocs={selectedDocs}
//         />
//         </div>
//     </Router>
//   );
// }

// export default App; 