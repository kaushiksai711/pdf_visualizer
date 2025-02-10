import React, { useState } from 'react';
import '../styles/DocumentSelection.css';

function DocumentSelection({ onDocsSelected }) {
  const [selectedFiles, setSelectedFiles] = useState([]);

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);

    const formData = new FormData();
    files.forEach(file => {
      formData.append('pdf', file);
    });

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      onDocsSelected(files.map((file, index) => ({
        id: index,
        name: file.name,
        path: data.files[index]
      })));
    } catch (error) {
      console.error('Error uploading files:', error);
    }
  };

  return (
    <div className="document-selection">
      <h2>Select Documents</h2>
      <div className="upload-area">
        <input
          type="file"
          multiple
          accept=".pdf"
          onChange={handleFileUpload}
          id="file-upload"
        />
        <label htmlFor="file-upload">
          Drop PDFs here or click to upload
        </label>
      </div>
      {selectedFiles.length > 0 && (
        <div className="selected-files">
          <h3>Selected Files:</h3>
          <ul>
            {selectedFiles.map((file, index) => (
              <li key={index}>{file.name}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default DocumentSelection; 