import React, { useState, useEffect, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { useParams } from 'react-router-dom';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import '../styles/PDFViewer.css';
// Set up PDF.js worker with proper error handling
const setupPdfWorker = () => {
  try {
    pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;
    return true;
  } catch (error) {
    console.error('Error setting up PDF.js worker:', error);
    return false;
  }
};

// Call the setup function
const workerSetupSuccess = setupPdfWorker();

const PDFViewer = ({ 
  isOpen, 
  onClose, 
  graphData, 
  selectedNodeTypes 
}) => {
  const { fileName } = useParams();
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1.2);
  const [highlights, setHighlights] = useState([]);
  const [pdfText, setPdfText] = useState([]);
  const [loading, setLoading] = useState(true);
  const [workerError, setWorkerError] = useState(!workerSetupSuccess);
  const [pdfUrl, setPdfUrl] = useState('');
  const [highlightColors, setHighlightColors] = useState({
    DOCUMENT: 'rgba(255, 255, 255, 0.3)',    // White (transparent)
    MAIN_TOPIC: 'rgba(77, 182, 80, 0.3)',    // Green
    SUBTOPIC: 'rgba(99, 175, 237, 0.3)',     // Blue
    CONCEPT: 'rgba(255, 193, 7, 0.3)',       // Yellow
    ENTITY: 'rgba(233, 30, 99, 0.3)'         // Pink
  });
  const [showColorLegend, setShowColorLegend] = useState(false);
  const [highlightOptionsVisible, setHighlightOptionsVisible] = useState(false);
  const [enabledHighlights, setEnabledHighlights] = useState({
    DOCUMENT: true,
    MAIN_TOPIC: true,
    SUBTOPIC: true,
    CONCEPT: true,
    ENTITY: true
  });
  const [textItems, setTextItems] = useState([]);
  
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (isOpen && fileName) {
      setLoading(true);
      fetchPdfText();
      
      // Generate PDF URL
      const url = `http://localhost:5000/api/pdf?file_name=${encodeURIComponent(fileName)}`;
      setPdfUrl(url);
    }
  }, [isOpen, fileName]);

  useEffect(() => {
    if (pdfText.length > 0 && graphData) {
      generateHighlights();
      console.log('Generated highlights:', highlights);
    }
  }, [pdfText, graphData, enabledHighlights]);

  useEffect(() => {
    console.log('Current highlights:', highlights);
    console.log('Enabled types:', enabledHighlights);
    console.log('Colors:', highlightColors);
  }, [highlights, enabledHighlights, highlightColors]);

  useEffect(() => {
    console.log('Current page highlights:', 
      highlights.filter(h => h.page === pageNumber)
    );
  }, [pageNumber, highlights]);

  useEffect(() => {
    if (highlights.length > 0) {
      console.log('Current highlights:', {
        total: highlights.length,
        byPage: highlights.reduce((acc, h) => {
          acc[h.page] = (acc[h.page] || 0) + 1;
          return acc;
        }, {}),
        types: highlights.reduce((acc, h) => {
          acc[h.type] = (acc[h.type] || 0) + 1;
          return acc;
        }, {})
      });
    }
  }, [highlights]);

  const fetchPdfText = async () => {
    try {
      // Fetch the extracted text from the PDF
      const response = await fetch(`http://localhost:5000/api/pdf-text?file_name=${encodeURIComponent(fileName)}`);
      if (!response.ok) throw new Error('Failed to fetch PDF text');
      
      const data = await response.json();
      setPdfText(data.pages || []);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching PDF text:', error);
      setLoading(false);
    }
  };

  const generateHighlights = () => {
    if (!graphData || !pdfText.length) return;
    
    const newHighlights = [];
    
    // Get all node labels from graph data, filtering by selected node types
    const nodeTerms = graphData.nodes
      .filter(node => enabledHighlights[node.type])
      .map(node => ({
        text: node.label,
        type: node.type,
        // Also include terms if available
        additionalTerms: node.terms || []
      }));
    
    // For each page of text
    pdfText.forEach((pageText, pageIndex) => {
      // Process each node and its terms
      nodeTerms.forEach(node => {
        // Find main label
        findOccurrences(pageText, node.text, pageIndex + 1, node.type, newHighlights);
        
        // Find additional terms
        node.additionalTerms.forEach(term => {
          findOccurrences(pageText, term, pageIndex + 1, node.type, newHighlights);
        });
      });
    });
    console.log(newHighlights,'adsadas');
    setHighlights(newHighlights);
  };

  const findOccurrences = (text, searchTerm, pageNum, nodeType, highlightArray) => {
    if (!searchTerm || searchTerm.length < 3) return; // Skip very short terms
    
    const searchTermLower = searchTerm.toLowerCase();
    const textLower = text.toLowerCase();
    let startPos = 0;
    
    while (startPos < textLower.length) {
      const foundPos = textLower.indexOf(searchTermLower, startPos);
      if (foundPos === -1) break;
      
      highlightArray.push({
        page: pageNum,
        text: searchTerm,
        position: foundPos,
        type: nodeType
      });
      
      startPos = foundPos + searchTermLower.length;
    }
  };

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setLoading(false);
    setWorkerError(false); // Worker is working if we got here
  };

  const onDocumentLoadError = (error) => {
    console.error('Error loading PDF:', error);
    setLoading(false);
    setWorkerError(true);
  };

  const onTextLayerRender = (textContent) => {
    // Store text items for better highlighting
    setTextItems(textContent.items);
  };

  const nextPage = () => {
    if (pageNumber < numPages) {
      setPageNumber(pageNumber + 1);
    }
  };

  const prevPage = () => {
    if (pageNumber > 1) {
      setPageNumber(pageNumber - 1);
    }
  };

  const zoomIn = () => {
    setScale(scale + 0.2);
  };

  const zoomOut = () => {
    if (scale > 0.5) {
      setScale(scale - 0.2);
    }
  };

  const downloadHighlightedPDF = async () => {
    try {
      // Show loading indicator
      setLoading(true);

      // Prepare highlight data for backend
      const highlightData = {
        fileName: fileName,
        highlights: highlights.filter(h => enabledHighlights[h.type]),
        nodeTypes: Object.keys(enabledHighlights).filter(type => enabledHighlights[type]),
        graphData: graphData  // Add the entire graph data
      };

      // Send highlights to backend
      const response = await fetch('http://localhost:5000/api/highlight-pdf', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(highlightData),
      });

      if (!response.ok) throw new Error('Failed to generate highlighted PDF');

      // Get the PDF blob
      const blob = await response.blob();
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `highlighted_${fileName}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      setLoading(false);
    } catch (error) {
      console.error('Error downloading highlighted PDF:', error);
      setLoading(false);
      alert('Failed to download highlighted PDF');
    }
  };
  const toggleHighlightType = (type) => {
    setEnabledHighlights(prev => ({
      ...prev,
      [type]: !prev[type]
    }));
  };

  // Improved text renderer using exact matching
 const enhancedTextRenderer = ({ str, itemIndex }) => {
    // Filter highlights for the current page and enabled types
    const pageHighlights = highlights.filter(h => 
      h.page === pageNumber && 
      enabledHighlights[h.type]
    );
    
    if (pageHighlights.length === 0) return str;
    
    let result = str;
    const segments = [];
    let lastIndex = 0;
    
    // Find all matches in the current text string
    pageHighlights.forEach(highlight => {
      const searchText = highlight.text.toLowerCase();
      const content = str.toLowerCase();
      let startIndex = 0;
      
      while (true) {
        const index = content.indexOf(searchText, startIndex);
        if (index === -1) break;
        
        segments.push({
          start: index,
          end: index + searchText.length,
          type: highlight.type,
          text: str.substring(index, index + searchText.length)
        });
        
        startIndex = index + searchText.length;
      }
    });
    
    if (segments.length === 0) return str;
    
    // Sort segments by start position
    segments.sort((a, b) => a.start - b.start);
    
    // Build the result with React elements
    const elements = [];
    segments.forEach((segment, index) => {
      // Add text before the highlight
      if (segment.start > lastIndex) {
        elements.push(str.substring(lastIndex, segment.start));
      }
      
      // Add the highlighted text
      elements.push(
        <mark 
          key={`highlight-${itemIndex}-${index}`}
          style={{
            backgroundColor: highlightColors[segment.type],
            color: 'inherit',
            padding: '0 2px',
            margin: '0 -2px',
            borderRadius: '2px',
            display: 'inline',
            position: 'relative',
            zIndex: 1
          }}
        >
          {segment.text}
        </mark>
      );
      
      lastIndex = segment.end;
    });
    
    // Add any remaining text
    if (lastIndex < str.length) {
      elements.push(str.substring(lastIndex));
    }
    console.log(elements);
    return elements;
  };

  if (!isOpen) return null;

  return (
    <div className="pdf-viewer-overlay" ref={containerRef}>
      <div className="pdf-viewer-container">
        <div className="pdf-viewer-header">
          <h3>PDF Viewer: {fileName}</h3>
          <div className="pdf-controls">
            <button onClick={prevPage} disabled={pageNumber <= 1}>
              Previous
            </button>
            <span>
              Page {pageNumber} of {numPages || '?'}
            </span>
            <button onClick={nextPage} disabled={pageNumber >= numPages}>
              Next
            </button>
            <button onClick={zoomIn}>Zoom +</button>
            <button onClick={zoomOut}>Zoom -</button>
            <button 
              onClick={() => setHighlightOptionsVisible(!highlightOptionsVisible)}
              className="highlight-options-btn"
            >
              Highlight Options
            </button>
            <button onClick={downloadHighlightedPDF} className="download-btn">
              Download Highlighted PDF
            </button>
            <button onClick={onClose} className="close-btn">
              Close
            </button>
          </div>
        </div>

        {highlightOptionsVisible && (
          <div className="highlight-options-panel">
            <h4>Highlight Options</h4>
            <div className="highlight-toggles">
              {Object.keys(highlightColors).map(type => (
                <label key={type} className="highlight-toggle">
                  <input
                    type="checkbox"
                    checked={enabledHighlights[type]}
                    onChange={() => toggleHighlightType(type)}
                  />
                  <span 
                    className="color-sample" 
                    style={{ backgroundColor: highlightColors[type].replace('0.3', '0.7') }}
                  ></span>
                  {type}
                </label>
              ))}
            </div>
          </div>
        )}

        {workerError ? (
          <div className="pdf-error">
            <p>Error loading PDF.js worker. Please try the following:</p>
            <ol>
              <li>Install pdfjs-dist: <code>npm install pdfjs-dist@3.6.172</code></li>
              <li>Configure webpack to handle the worker file properly</li>
              <li>Check the console for specific error messages</li>
            </ol>
          </div>
        ) : loading ? (
          <div className="pdf-loading">
            <div className="spinner"></div>
            <span>Loading PDF and generating highlights...</span>
          </div>
        ) : (
          <div className="pdf-content" ref={canvasRef}>
            {pdfUrl && (
              <Document
                file={pdfUrl}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
              >
                <Page
                  pageNumber={pageNumber}
                  scale={scale}
                  renderTextLayer={true}
                  renderAnnotationLayer={true}
                  onGetTextSuccess={onTextLayerRender}
                  customTextRenderer={enhancedTextRenderer}
                  className="pdf-page"
                  loading={<div>Loading page...</div>}
                  error={<div>Error loading page!</div>}
                />
              </Document>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default PDFViewer;