"use client"

import React, { useState } from 'react'
import '../styles/DocumentPanel.css'

function DocumentPanel({ activeInstance, setActiveInstance, selectedDocs, setSelectedDocs }) {
  const [instances, setInstances] = useState([])
  const [editingId, setEditingId] = useState(null)

  const createNewInstance = () => {
    const newInstance = {
      id: Date.now(),
      name: `Instance ${instances.length + 1}`,
      documents: [],
      sessionId: null,
      messages: [],
      relevantChunks: []
    }
    const updatedInstances = [...instances, newInstance]
    setInstances(updatedInstances)
    setActiveInstance(newInstance)
  }

  return (
    <div className="panel document-panel">
      <div className="panel-header">
        <h2>Chat Instances</h2>
      </div>
      <div className="panel-content">
        <button className="new-instance-btn" onClick={createNewInstance}>
          New Instance
        </button>

        <div className="instances-list">
          {instances.map(instance => (
            <div 
              key={instance.id} 
              className={`instance-item ${activeInstance?.id === instance.id ? 'active' : ''}`}
              onClick={() => setActiveInstance(instance)}
            >
              {editingId === instance.id ? (
                <input
                  type="text"
                  value={instance.name}
                  onChange={(e) => {
                    setInstances(instances.map(inst =>
                      inst.id === instance.id ? { ...inst, name: e.target.value } : inst
                    ))
                  }}
                  onBlur={() => setEditingId(null)}
                  autoFocus
                />
              ) : (
                <div 
                  className="instance-name"
                  onDoubleClick={() => setEditingId(instance.id)}
                >
                  {instance.name}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default DocumentPanel
// "use client"

// import React, { useState } from 'react'
// import '../styles/DocumentPanel.css'

// function DocumentPanel({ activeInstance, setActiveInstance, selectedDocs, setSelectedDocs }) {
//   const [instances, setInstances] = useState([])
//   const [editingId, setEditingId] = useState(null)

//   const createNewInstance = () => {
//     const newInstance = {
//       id: Date.now(),
//       name: `Instance ${instances.length + 1}`,
//       documents: [],
//       sessionId: null,
//       messages: [],
//       relevantChunks: []
//     }
//     const updatedInstances = [...instances, newInstance]
//     setInstances(updatedInstances)
//     setActiveInstance(newInstance)
//   }

//   return (
//     <div className="panel document-panel">
//       <div className="panel-header">
//         <h2>Chat Instances</h2>
//       </div>
//       <div className="panel-content">
//         <button className="new-instance-btn" onClick={createNewInstance}>
//           New Instance
//         </button>

//         <div className="instances-list">
//           {instances.map(instance => (
//             <div 
//               key={instance.id} 
//               className={`instance-item ${activeInstance?.id === instance.id ? 'active' : ''}`}
//               onClick={() => setActiveInstance(instance)}
//             >
//               {editingId === instance.id ? (
//                 <input
//                   type="text"
//                   value={instance.name}
//                   onChange={(e) => {
//                     setInstances(instances.map(inst =>
//                       inst.id === instance.id ? { ...inst, name: e.target.value } : inst
//                     ))
//                   }}
//                   onBlur={() => setEditingId(null)}
//                   autoFocus
//                 />
//               ) : (
//                 <div 
//                   className="instance-name"
//                   onDoubleClick={() => setEditingId(instance.id)}
//                 >
//                   {instance.name}
//                 </div>
//               )}
//             </div>
//           ))}
//         </div>
//       </div>
//     </div>
//   )
// }

// export default DocumentPanel
// import React, { useState } from 'react';
// import '../styles/DocumentPanel.css';
// import KnowledgeGraph from './KnowledgeGraph';
// import { useNavigate } from 'react-router-dom';

// function DocumentPanel({ activeInstance, setActiveInstance, selectedDocs, setSelectedDocs }) {
//   const [instances, setInstances] = useState([]);
//   const [editingId, setEditingId] = useState(null);
//   const [showGraph, setShowGraph] = useState(false);
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [uploadProgress, setUploadProgress] = useState({});
//   const navigate = useNavigate();

//   const createNewInstance = () => {
//     const newInstance = {
//       id: Date.now(),
//       name: `Instance ${instances.length + 1}`,
//       documents: [],
//       sessionId: null,
//       messages: [],
//       relevantChunks: []
//     };
//     setInstances([...instances, newInstance]);
//     setActiveInstance(newInstance);
//   };

//   const handleFileUpload = async (event) => {
//     const files = Array.from(event.target.files);
//     const CHUNK_SIZE = 1024 * 1024; // 1MB chunks

//     for (const file of files) {
//       try {
//         const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
//         const fileId = Date.now().toString();
        
//         // Start upload session
//         await fetch('http://localhost:5000/api/upload/start', {
//           method: 'POST',
//           headers: { 'Content-Type': 'application/json' },
//           body: JSON.stringify({
//             fileName: file.name,
//             fileSize: file.size,
//             fileId: fileId,
//             totalChunks: totalChunks
//           })
//         });

//         // Upload chunks
//         for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
//           const start = chunkIndex * CHUNK_SIZE;
//           const end = Math.min(start + CHUNK_SIZE, file.size);
//           const chunk = file.slice(start, end);

//           const formData = new FormData();
//           formData.append('chunk', chunk);
//           formData.append('fileId', fileId);
//           formData.append('chunkIndex', chunkIndex);

//           await fetch('http://localhost:5000/api/upload/chunk', {
//             method: 'POST',
//             body: formData
//           });

//           setUploadProgress(prev => ({
//             ...prev,
//             [file.name]: {
//               total: totalChunks,
//               current: chunkIndex + 1
//             }
//           }));
//         }

//         // Complete upload
//         const response = await fetch('http://localhost:5000/api/upload/complete', {
//           method: 'POST',
//           headers: { 'Content-Type': 'application/json' },
//           body: JSON.stringify({ 
//             fileId: fileId,
//             fileName: file.name 
//           })
//         });

//         const data = await response.json();
        
//         // Update UI with new document
//         const newDoc = {
//           id: fileId,
//           name: file.name,
//           path: data.path
//         };

//         setSelectedDocs(prev => [...prev, newDoc]);
        
//         if (activeInstance) {
//           setInstances(prev => prev.map(inst => 
//             inst.id === activeInstance.id 
//               ? { ...inst, documents: [...inst.documents, newDoc] }
//               : inst
//           ));
//         }

//         // Clear progress for this file
//         setUploadProgress(prev => {
//           const newProgress = { ...prev };
//           delete newProgress[file.name];
//           return newProgress;
//         });

//       } catch (error) {
//         console.error('Error uploading file:', error);
//         // Show error to user
//         alert(`Error uploading ${file.name}: ${error.message}`);
//       }
//     }
//   };

//   const viewKnowledgeGraph = (fileName) => {
//     console.log(fileName)
//     navigate(`/knowledge-graph/${fileName}`);
//   };

//   return (
//     <div className="panel document-panel">
//       <div className="panel-header">
//         <h2>Chat Instances</h2>
//       </div>
//       <div className="panel-content">
//         <button className="new-instance-btn" onClick={createNewInstance}>
//           New Instance
//         </button>

//         <div className="instances-list">
//           {instances.map(instance => (
//             <div 
//               key={instance.id} 
//               className={`instance-item ${activeInstance?.id === instance.id ? 'active' : ''}`}
//               onClick={() => setActiveInstance(instance)}
//             >
//               {editingId === instance.id ? (
//                 <input
//                   type="text"
//                   value={instance.name}
//                   onChange={(e) => {
//                     setInstances(instances.map(inst =>
//                       inst.id === instance.id ? { ...inst, name: e.target.value } : inst
//                     ));
//                   }}
//                   onBlur={() => setEditingId(null)}
//                   autoFocus
//                 />
//               ) : (
//                 <div 
//                   className="instance-name"
//                   onDoubleClick={() => setEditingId(instance.id)}
//                 >
//                   {instance.name}
//                 </div>
//               )}
//               <div className="instance-docs">
//                 {instance.documents.map(doc => (
//                   <div key={doc.id} className="doc-item">
//                     {doc.name}
//                     <button 
//                       onClick={(e) => {
//                         e.stopPropagation();
//                         viewKnowledgeGraph(doc.name);
//                       }}
//                       className="view-graph-btn"
//                     >
//                       View Knowledge Graph
//                     </button>
//                   </div>
//                 ))}
//               </div>
//             </div>
//           ))}
//         </div>

//         <div className="upload-section flex flex-col items-center gap-3 p-4 bg-white rounded-lg shadow-md">
//   {/* Upload Button */}
//   <label
//     htmlFor="file-upload"
//     className="px-4 py-2 bg-blue-600 text-white font-bold rounded-lg shadow-md hover:bg-blue-700 transition cursor-pointer"
//   >
//     📂 Upload File
//   </label>
//   <input
//     type="file"
//     onChange={handleFileUpload}
//     multiple
//     accept=".pdf"
//     id="file-upload"
//     className="hidden"
//   />

//   {/* Progress Bars */}
//   <div className="w-full">
//     {Object.entries(uploadProgress).map(([fileName, progress]) => (
//       <div key={fileName} className="upload-progress mt-2">
//         <div className="font-semibold text-gray-700">{fileName}</div>
//         <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden mt-1">
//           <div
//             className="bg-green-500 h-3 transition-all"
//             style={{ width: `${(progress.current / progress.total) * 100}%` }}
//           />
//         </div>
//         <div className="text-sm text-gray-600 mt-1">
//           {Math.round((progress.current / progress.total) * 100)}%
//         </div>
//       </div>
//     ))}
//   </div>
// </div>

//       </div>

//       {
//       showGraph && selectedFile && (
//         <KnowledgeGraph 
//           fileName={selectedFile}
//           onClose={() => {
//             console.log('asaasaa')
//             setShowGraph(false);
//             setSelectedFile(null);
//           }}
//         />
//       )}
//     </div>
//   );
// }

// export default DocumentPanel; 