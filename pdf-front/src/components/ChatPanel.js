"use client"

import React, { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import '../styles/ChatPanel.css'
import RagVisualizer from './RagVisualizer' // Make sure to import RagVisualizer
import TypewriterResponse from './TypeScriptWriter'
import '../styles/TypwWriterResponse.css'
function ChatPanel({ instance, updateInstanceData, onRemoveDoc }) {
  const [input, setInput] = useState('')
  const [uploadProgress, setUploadProgress] = useState({})
  const [showVisualizer, setShowVisualizer] = useState(false)
  const [visualizerExpanded, setVisualizerExpanded] = useState(false)
  const chatEndRef = useRef(null)
  const navigate = useNavigate()
  
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [instance?.messages])

  const handleFileUpload = async (event) => {
    if (!instance) return
    
    const files = Array.from(event.target.files)
    const CHUNK_SIZE = 1024 * 1024 // 1MB chunks

    for (const file of files) {
      try {
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE)
        const fileId = Date.now().toString()
        
        // Start upload session
        await fetch('http://localhost:5000/api/upload/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            fileName: file.name,
            fileSize: file.size,
            fileId: fileId,
            totalChunks: totalChunks
          })
        })

        // Upload chunks
        for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
          const start = chunkIndex * CHUNK_SIZE
          const end = Math.min(start + CHUNK_SIZE, file.size)
          const chunk = file.slice(start, end)

          const formData = new FormData()
          formData.append('chunk', chunk)
          formData.append('fileId', fileId)
          formData.append('chunkIndex', chunkIndex)

          await fetch('http://localhost:5000/api/upload/chunk', {
            method: 'POST',
            body: formData
          })

          setUploadProgress(prev => ({
            ...prev,
            [file.name]: {
              total: totalChunks,
              current: chunkIndex + 1
            }
          }))
        }

        // Complete upload
        const response = await fetch('http://localhost:5000/api/upload/complete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            fileId: fileId,
            fileName: file.name 
          })
        })

        const data = await response.json()
        
        // Update UI with new document
        const newDoc = {
          id: fileId,
          name: file.name,
          path: data.path
        }
        
        // Add document to instance
        const updatedDocs = [...(instance.documents || []), newDoc]
        instance.documents = updatedDocs
        
        // Clear progress for this file
        setUploadProgress(prev => {
          const newProgress = { ...prev }
          delete newProgress[file.name]
          return newProgress
        })

      } catch (error) {
        console.error('Error uploading file:', error)
        // Show error to user
        alert(`Error uploading ${file.name}: ${error.message}`)
      }
    }
  }

  const viewKnowledgeGraph = (fileName) => {
    navigate(`/knowledge-graph/${fileName}`)
  }

  const toggleVisualizer = () => {
    setShowVisualizer(prev => !prev)
  }

  const handleVisualizerToggleExpand = () => {
    setVisualizerExpanded(prev => !prev)
  }

  const handleSendMessage = async () => {
    if (!input.trim() || !instance || !instance.documents?.length) return

    const newMessage = { type: 'user', content: input }
    const updatedMessages = [...(instance.messages || []), newMessage]
    
    updateInstanceData(
      instance.sessionId, 
      updatedMessages,
      instance.relevantChunks
    )
    
    setInput('')
    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: instance.sessionId,
          query: input,
          selected_files: instance.documents.map(doc => doc.name)
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      if (!instance.sessionId && data.session_id) {
        updateInstanceData(
          data.session_id,
          [...updatedMessages, { type: 'bot', content: data.response }],
          data.relevant_chunks || []
        )
      } else {
        updateInstanceData(
          instance.sessionId,
          [...updatedMessages, { type: 'bot', content: data.response }],
          data.relevant_chunks || []
        )
      }
    } catch (error) {
      console.error('Error:', error)
      updateInstanceData(
        instance.sessionId,
        [...updatedMessages, { 
          type: 'bot', 
          content: 'Error: Failed to get response from server.' 
        }],
        instance.relevantChunks
      )
    }
  }

  const handleSummarize = async (chunk) => {
    try {
      const response = await fetch('http://localhost:5000/api/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: chunk.content })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      const updatedChunks = instance.relevantChunks.map(c => 
        c === chunk ? { ...c, summary: data.summary } : c
      )
      
      updateInstanceData(
        instance.sessionId, 
        instance.messages, 
        updatedChunks
      )
    } catch (error) {
      console.error('Error summarizing chunk:', error)
    }
  }

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
    )
  }

  return (
    <div className="panel-container" style={{ width: '100%' }}>
      <div 
        className="panel chat-panel" 
        style={{ 
          background: 'rgba(0, 0, 0, 0)', 
          borderRadius: '10px', 
          boxShadow: '0 0 20px rgba(255, 255, 255, 0.5)',
          width: showVisualizer ? '50%' : '100%'
        }}
      >      <div className="panel-header">
          <h2 style={{ color: '#ffffff' }}>Chat with Documents - {instance.name}</h2>
          <div className="panel-actions">
            {instance.documents?.length > 0 && (
              <button 
                onClick={toggleVisualizer}
                className="visualize-btn"
                style={{
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  padding: '8px 12px',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  marginLeft: '10px'
                }}
              >
                {showVisualizer ? 'Hide Visualizer' : 'Visualize'}
              </button>
            )}
          </div>
        </div>
        
        <div className="panel-content">
          {!instance.documents?.length ? (
            <div className="no-documents-message" style={{ color: '#ffffff', textAlign: 'center' }}>
              <p>No files uploaded yet</p>
              
              <div className="upload-section flex flex-col items-center gap-3 p-4 bg-white rounded-lg shadow-md mt-4">
                <label
                  htmlFor="file-upload"
                  className="px-4 py-2 bg-blue-600 text-white font-bold rounded-lg shadow-md hover:bg-blue-700 transition cursor-pointer"
                >
                  📂 Upload File
                </label>
                <input
                  type="file"
                  onChange={handleFileUpload}
                  multiple
                  accept=".pdf"
                  id="file-upload"
                  className="hidden"
                />

                {/* Progress Bars */}
                <div className="w-full">
                  {Object.entries(uploadProgress).map(([fileName, progress]) => (
                    <div key={fileName} className="upload-progress mt-2">
                      <div className="font-semibold text-gray-700">{fileName}</div>
                      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden mt-1">
                        <div
                          className="bg-green-500 h-3 transition-all"
                          style={{ width: `${(progress.current / progress.total) * 100}%` }}
                        />
                      </div>
                      <div className="text-sm text-gray-600 mt-1">
                        {Math.round((progress.current / progress.total) * 100)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <>
              <div className="selected-docs">
                {instance.documents.map(doc => (
                  <div key={doc.id} className="doc-tag">
                    {doc.name}
                    <button 
                      onClick={() => viewKnowledgeGraph(doc.name)}
                      className="view-graph-btn"
                      style={{color :'aquamarine'}}
                    >
                      View Graph
                    </button>
                    <button onClick={() => onRemoveDoc(doc.id)}>&times;</button>
                  </div>
                ))}
                
                <label
                  htmlFor="add-file-upload"
                  className="add-file-btn"
                >
                  + Add File
                </label>
                <input
                  type="file"
                  onChange={handleFileUpload}
                  multiple
                  accept=".pdf"
                  id="add-file-upload"
                  className="hidden"
                />
              </div>
              
              {/* <div className="chat-messages">
                {instance.messages?.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.type}`}>
                    {msg.content}
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div> */}
              {/* <div className="chat-messages">
              {instance.messages?.map((msg, idx) => (
                <div key={idx} className={`message ${msg.type}`}>
                  {msg.type === 'bot' ? (
                    <TypewriterResponse text={msg.content} speed={5} />
                  ) : (
                    msg.content
                  )}
                </div>
              ))}
              <div ref={chatEndRef} />
            </div> */}
            <div className="chat-messages">
                {instance.messages?.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.type}`}>
                    {msg.type === 'bot' ? (
                      <TypewriterResponse text={msg.content} speed={5} />
                    ) : (
                      msg.content
                    )}
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
            </>
          )}
        </div>

        <div className="chat-input">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder={instance.documents?.length ? "Ask a question..." : "Please upload documents first"}
            disabled={!instance.documents?.length}
          />
          <button 
            onClick={handleSendMessage}
            disabled={!instance.documents?.length}
          >
            Send
          </button>
        </div>
      </div>

      {showVisualizer && instance.documents?.length > 0 && (
      <RagVisualizer 
        isExpanded={visualizerExpanded} 
        onToggleExpand={handleVisualizerToggleExpand}
        selectedDocs={instance.documents}
      />
    )}
    </div>
  )
}

export default ChatPanel
// "use client"

// import React, { useState, useRef, useEffect } from 'react'
// import { useNavigate } from 'react-router-dom'
// import '../styles/ChatPanel.css'

// function ChatPanel({ instance, updateInstanceData, onRemoveDoc }) {
//   const [input, setInput] = useState('')
//   const [uploadProgress, setUploadProgress] = useState({})
//   const chatEndRef = useRef(null)
//   const navigate = useNavigate()
  
//   useEffect(() => {
//     chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
//   }, [instance?.messages])

//   const handleFileUpload = async (event) => {
//     if (!instance) return
    
//     const files = Array.from(event.target.files)
//     const CHUNK_SIZE = 1024 * 1024 // 1MB chunks

//     for (const file of files) {
//       try {
//         const totalChunks = Math.ceil(file.size / CHUNK_SIZE)
//         const fileId = Date.now().toString()
        
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
//         })

//         // Upload chunks
//         for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
//           const start = chunkIndex * CHUNK_SIZE
//           const end = Math.min(start + CHUNK_SIZE, file.size)
//           const chunk = file.slice(start, end)

//           const formData = new FormData()
//           formData.append('chunk', chunk)
//           formData.append('fileId', fileId)
//           formData.append('chunkIndex', chunkIndex)

//           await fetch('http://localhost:5000/api/upload/chunk', {
//             method: 'POST',
//             body: formData
//           })

//           setUploadProgress(prev => ({
//             ...prev,
//             [file.name]: {
//               total: totalChunks,
//               current: chunkIndex + 1
//             }
//           }))
//         }

//         // Complete upload
//         const response = await fetch('http://localhost:5000/api/upload/complete', {
//           method: 'POST',
//           headers: { 'Content-Type': 'application/json' },
//           body: JSON.stringify({ 
//             fileId: fileId,
//             fileName: file.name 
//           })
//         })

//         const data = await response.json()
        
//         // Update UI with new document
//         const newDoc = {
//           id: fileId,
//           name: file.name,
//           path: data.path
//         }
        
//         // Add document to instance
//         const updatedDocs = [...(instance.documents || []), newDoc]
//         instance.documents = updatedDocs
        
//         // Clear progress for this file
//         setUploadProgress(prev => {
//           const newProgress = { ...prev }
//           delete newProgress[file.name]
//           return newProgress
//         })

//       } catch (error) {
//         console.error('Error uploading file:', error)
//         // Show error to user
//         alert(`Error uploading ${file.name}: ${error.message}`)
//       }
//     }
//   }

//   const viewKnowledgeGraph = (fileName) => {
//     navigate(`/knowledge-graph/${fileName}`)
//   }

//   const handleSendMessage = async () => {
//     if (!input.trim() || !instance || !instance.documents?.length) return

//     const newMessage = { type: 'user', content: input }
//     const updatedMessages = [...(instance.messages || []), newMessage]
    
//     updateInstanceData(
//       instance.sessionId, 
//       updatedMessages,
//       instance.relevantChunks
//     )
    
//     setInput('')
//     try {
//       const response = await fetch('http://localhost:5000/api/chat', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({
//           session_id: instance.sessionId,
//           query: input,
//           selected_files: instance.documents.map(doc => doc.name)
//         })
//       })

//       if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`)
//       }

//       const data = await response.json()
      
//       if (!instance.sessionId && data.session_id) {
//         updateInstanceData(
//           data.session_id,
//           [...updatedMessages, { type: 'bot', content: data.response }],
//           data.relevant_chunks || []
//         )
//       } else {
//         updateInstanceData(
//           instance.sessionId,
//           [...updatedMessages, { type: 'bot', content: data.response }],
//           data.relevant_chunks || []
//         )
//       }
//     } catch (error) {
//       console.error('Error:', error)
//       updateInstanceData(
//         instance.sessionId,
//         [...updatedMessages, { 
//           type: 'bot', 
//           content: 'Error: Failed to get response from server.' 
//         }],
//         instance.relevantChunks
//       )
//     }
//   }

//   const handleSummarize = async (chunk) => {
//     try {
//       const response = await fetch('http://localhost:5000/api/summarize', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ text: chunk.content })
//       })

//       if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`)
//       }

//       const data = await response.json()
//       const updatedChunks = instance.relevantChunks.map(c => 
//         c === chunk ? { ...c, summary: data.summary } : c
//       )
      
//       updateInstanceData(
//         instance.sessionId, 
//         instance.messages, 
//         updatedChunks
//       )
//     } catch (error) {
//       console.error('Error summarizing chunk:', error)
//     }
//   }

//   if (!instance) {
//     return (
//       <div className="panel chat-panel">
//         <div className="panel-header">
//           <h2>Chat with Documents</h2>
//         </div>
//         <div className="panel-content">
//           <div className="no-instance-message">
//             Please select or create a new instance to start chatting
//           </div>
//         </div>
//       </div>
//     )
//   }

//   return (
//     <div className="panel chat-panel" style={{ background: 'rgba(0, 0, 0, 0.7)', borderRadius: '10px', boxShadow: '0 0 20px rgba(255, 255, 255, 0.5)' }}>
//       <div className="panel-header">
//         <h2 style={{ color: '#ffffff' }}>Chat with Documents - {instance.name}</h2>
//       </div>
      
//       <div className="panel-content">
//         {!instance.documents?.length ? (
//           <div className="no-documents-message" style={{ color: '#ffffff', textAlign: 'center' }}>
//             <p>if no file uploaded</p>
//             <p>I want a upload box</p>
//             <p>I want the uploaded files for this instance</p>
//             <p>be displayed here with an options</p>
//             <p>to open the file and knowledge graph</p>
            
//             <div className="upload-section flex flex-col items-center gap-3 p-4 bg-white rounded-lg shadow-md mt-4">
//               <label
//                 htmlFor="file-upload"
//                 className="px-4 py-2 bg-blue-600 text-white font-bold rounded-lg shadow-md hover:bg-blue-700 transition cursor-pointer"
//               >
//                 📂 Upload File
//               </label>
//               <input
//                 type="file"
//                 onChange={handleFileUpload}
//                 multiple
//                 accept=".pdf"
//                 id="file-upload"
//                 className="hidden"
//               />

//               {/* Progress Bars */}
//               <div className="w-full">
//                 {Object.entries(uploadProgress).map(([fileName, progress]) => (
//                   <div key={fileName} className="upload-progress mt-2">
//                     <div className="font-semibold text-gray-700">{fileName}</div>
//                     <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden mt-1">
//                       <div
//                         className="bg-green-500 h-3 transition-all"
//                         style={{ width: `${(progress.current / progress.total) * 100}%` }}
//                       />
//                     </div>
//                     <div className="text-sm text-gray-600 mt-1">
//                       {Math.round((progress.current / progress.total) * 100)}%
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             </div>
//           </div>
//         ) : (
//           <>
//             <div className="selected-docs">
//               {instance.documents.map(doc => (
//                 <div key={doc.id} className="doc-tag">
//                   {doc.name}
//                   <button 
//                     onClick={() => viewKnowledgeGraph(doc.name)}
//                     className="view-graph-btn"
//                     style={{color :'aquamarine'}}
//                   >
//                     View Graph
//                   </button>
//                   <button onClick={() => onRemoveDoc(doc.id)}>&times;</button>
//                 </div>
//               ))}
              
//               <label
//                 htmlFor="add-file-upload"
//                 className="add-file-btn"
//               >
//                 + Add File
//               </label>
//               <input
//                 type="file"
//                 onChange={handleFileUpload}
//                 multiple
//                 accept=".pdf"
//                 id="add-file-upload"
//                 className="hidden"
//               />
//             </div>
            
//             <div className="chat-messages">
//               {instance.messages?.map((msg, idx) => (
//                 <div key={idx} className={`message ${msg.type}`}>
//                   {msg.content}
//                 </div>
//               ))}
//               <div ref={chatEndRef} />
//             </div>

//             <div className="relevant-chunks">
//               {instance.relevantChunks?.map((chunk, idx) => (
//                 <div key={idx} className="chunk">
//                   <div className="chunk-content">{chunk.content}</div>
//                   {chunk.summary ? (
//                     <div className="chunk-summary">{chunk.summary}</div>
//                   ) : (
//                     <button 
//                       className="summarize-btn"
//                       onClick={() => handleSummarize(chunk)}
//                     >
//                       Summarize
//                     </button>
//                   )}
//                 </div>
//               ))}
//             </div>
//           </>
//         )}
//       </div>

//       <div className="chat-input">
//         <input
//           type="text"
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//           onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
//           placeholder={instance.documents?.length ? "Ask a question..." : "Please upload documents first"}
//           disabled={!instance.documents?.length}
//         />
//         <button 
//           onClick={handleSendMessage}
//           disabled={!instance.documents?.length}
//         >
//           Send
//         </button>
//       </div>
//     </div>
//   )
// }

// export default ChatPanel
// import React, { useState, useRef, useEffect } from 'react';
// import '../styles/ChatPanel.css';

// function ChatPanel({ instance, updateInstanceData, onRemoveDoc }) {
//   const [input, setInput] = useState('');
//   const chatEndRef = useRef(null);
//   console.log(instance)
//   useEffect(() => {
//     chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   }, [instance?.messages]);

//   const handleSendMessage = async () => {
//     if (!input.trim() || !instance || !instance.documents.length) return;

//     const newMessage = { type: 'user', content: input };
//     const updatedMessages = [...(instance.messages || []), newMessage];
    
//     updateInstanceData(
//       instance.sessionId, 
//       updatedMessages,
//       instance.relevantChunks
//     );
    
//     setInput('');
//     try {
//       const response = await fetch('http://localhost:5000/api/chat', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({
//           session_id: instance.sessionId,
//           query: input,
//           selected_files:instance.documents.map(doc => doc.name)// Using first document for now
//         })
//       });

//       if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//       }

//       const data = await response.json();
      
//       if (!instance.sessionId && data.session_id) {
//         updateInstanceData(
//           data.session_id,
//           [...updatedMessages, { type: 'bot', content: data.response }],
//           data.relevant_chunks || []
//         );
//       } else {
//         updateInstanceData(
//           instance.sessionId,
//           [...updatedMessages, { type: 'bot', content: data.response }],
//           data.relevant_chunks || []
//         );
//       }
//     } catch (error) {
//       console.error('Error:', error);
//       updateInstanceData(
//         instance.sessionId,
//         [...updatedMessages, { 
//           type: 'bot', 
//           content: 'Error: Failed to get response from server.' 
//         }],
//         instance.relevantChunks
//       );
//     }
//   };

//   const handleSummarize = async (chunk) => {
//     try {
//       const response = await fetch('http://localhost:5000/api/summarize', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ text: chunk.content })
//       });

//       if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//       }

//       const data = await response.json();
//       const updatedChunks = instance.relevantChunks.map(c => 
//         c === chunk ? { ...c, summary: data.summary } : c
//       );
      
//       updateInstanceData(instance.sessionId, instance.messages, updatedChunks);
//     } catch (error) {
//       console.error('Error summarizing chunk:', error);
//     }
//   };

//   if (!instance) {
//     return (
//       <div className="panel chat-panel">
//         <div className="panel-header">
//           <h2>Chat with Documents</h2>
//         </div>
//         <div className="panel-content">
//           <div className="no-instance-message">
//             Please select or create a new instance to start chatting
//           </div>
//         </div>
//       </div>
//     );
//   }

//   return (
//     <div className="panel chat-panel" style={{ background: 'rgba(0, 0, 0, 0.7)', borderRadius: '10px', boxShadow: '0 0 20px rgba(255, 255, 255, 0.5)' }}>
//       <div className="panel-header">
//         <h2 style={{ color: '#ffffff' }}>Chat with Documents - {instance.name}</h2>
//         <div className="selected-docs">
//           {instance.documents.map(doc => (
//             <div key={doc.id} className="doc-tag">
//               {doc.name}
//               <button onClick={() => onRemoveDoc(doc.id)}>&times;</button>
//             </div>
//           ))}
//         </div>
//       </div>
      
//       <div className="panel-content">
//         <div className="chat-messages">
//           {instance.messages?.map((msg, idx) => (
//             <div key={idx} className={`message ${msg.type}`}>
//               {msg.content}
//             </div>
//           ))}
//           <div ref={chatEndRef} />
//         </div>

//         <div className="relevant-chunks">
//           {instance.relevantChunks?.map((chunk, idx) => (
//             <div key={idx} className="chunk">
//               <div className="chunk-content">{chunk.content}</div>
//               {chunk.summary ? (
//                 <div className="chunk-summary">{chunk.summary}</div>
//               ) : (
//                 <button 
//                   className="summarize-btn"
//                   onClick={() => handleSummarize(chunk)}
//                 >
//                   Summarize
//                 </button>
//               )}
//             </div>
//           ))}
//         </div>
//       </div>

//       <div className="chat-input">
//         <input
//           type="text"
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//           onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
//           placeholder={instance.documents.length ? "Ask a question..." : "Please upload documents first"}
//           disabled={!instance.documents.length}
//         />
//         <button 
//           onClick={handleSendMessage}
//           disabled={!instance.documents.length}
//         >
//           Send
//         </button>
//       </div>
//     </div>
//   );
// }

// export default ChatPanel; 