import React, { useEffect, useRef, useState } from 'react';
import PDFViewer from './PDFViewer';
import ForceGraph3D from '3d-force-graph';
import * as THREE from 'three';
import { 
  forceSimulation,
  forceManyBody,
  forceLink,
  forceCenter,
  forceY,
  forceCollide
} from 'd3-force-3d';
import '../styles/KnowledgeGraph.css';
import { useNavigate, useParams } from 'react-router-dom';
const api_key="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"

// Updated color scheme for better visibility - moved to the top level
const getNodeColor = (type) => {
  switch (type) {
    case 'DOCUMENT':
      return '#FFFFFF'; // White for document root
    case 'MAIN_TOPIC':
      return '#4CAF50'; // Bright green
    case 'SUBTOPIC':
      return '#2196F3'; // Bright blue
    case 'CONCEPT':
      return '#FFC107'; // Bright yellow
    case 'ENTITY':
      return '#E91E63'; // Bright pink
    default:
      return '#FF5722'; // Bright orange
  }
};

// Helper function to get connected nodes - moved to the top level
const getConnectedNodes = (nodeId, graphData) => {
  if (!graphData) return [];
  
  const connectedNodes = [];
  graphData.links.forEach(link => {
    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
    const targetId = typeof link.target === 'object' ? link.target.id : link.target;
    
    if (sourceId === nodeId) {
      const targetNode = graphData.nodes.find(node => node.id === targetId);
      if (targetNode) {
        connectedNodes.push({
          ...targetNode,
          relationship: link.relationship || 'connected to'
        });
      }
    } else if (targetId === nodeId) {
      const sourceNode = graphData.nodes.find(node => node.id === sourceId);
      if (sourceNode) {
        connectedNodes.push({
          ...sourceNode,
          relationship: (link.relationship ? 'reverse ' + link.relationship : 'connected to')
        });
      }
    }
  });
  
  return connectedNodes;
};

// Custom SpriteText class (existing code)
class SpriteText extends THREE.Sprite {
  constructor(text, textHeight = 8, color = '#ffffff', backgroundColor = 'rgba(0,0,0,0.8)') {
    super(new THREE.SpriteMaterial({ map: new THREE.Texture() }));
    this.text = text;
    this.textHeight = textHeight;
    this.color = color;
    this.backgroundColor = backgroundColor;
    this.padding = 2;
    this.updateTexture();
  }

  updateTexture() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.font = `${this.textHeight}px Arial`;
    
    const textWidth = ctx.measureText(this.text).width;
    canvas.width = textWidth + (this.padding * 2);
    canvas.height = this.textHeight + (this.padding * 2);
    
    // Draw background
    ctx.fillStyle = this.backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw text
    ctx.font = `${this.textHeight}px Arial`;
    ctx.fillStyle = this.color;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    ctx.fillText(this.text, canvas.width / 2, canvas.height / 2);
    
    const texture = new THREE.Texture(canvas);
    texture.needsUpdate = true;
    this.material.map = texture;
    this.scale.set(canvas.width / canvas.height * this.textHeight, this.textHeight, 1);
  }
}

// New component for connected topics
const ConnectedTopicsList = ({ topics, onTopicClick }) => {
  return (
    <div className="connected-topics">
      <h4>Connected Topics</h4>
      {topics.length > 0 ? (
        <ul>
          {topics.map((topic, index) => (
            <li key={index} onClick={() => onTopicClick(topic)}>
              <span className="topic-type" style={{ backgroundColor: getNodeColor(topic.type) }}></span>
              <span className="topic-label">{topic.label}</span>
              <span className="topic-type-label">({topic.type})</span>
            </li>
          ))}
        </ul>
      ) : (
        <p>No connected topics found</p>
      )}
    </div>
  );
};

// Node state visualization component
const NodeStateVisualization = () => {
  const nodeTypes = [
    { type: 'DOCUMENT', label: 'Document' },
    { type: 'MAIN_TOPIC', label: 'Main Topic' },
    { type: 'SUBTOPIC', label: 'Subtopic' },
    { type: 'CONCEPT', label: 'Concept' },
    { type: 'ENTITY', label: 'Entity' }
  ];

  return (
    <div className="node-state">
      <div>Node Types:</div>
      {nodeTypes.map((item, index) => (
        <div key={index} className="node-state-item">
          <div 
            className="node-state-color" 
            style={{ backgroundColor: getNodeColor(item.type) }}
          ></div>
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  );
};

// Chat widget component
const ChatWidget = ({ selectedNode, graphData }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Animation for opening/closing
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.style.opacity = isOpen ? 1 : 0.95;
    }
  }, [isOpen]);

  // Send message to OpenRouter API
  const sendMessage = async () => {
    if (!inputValue.trim()) return;
    
    // Add user message to chat
    const userMessage = {
      sender: 'user',
      text: inputValue,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Create context about the graph and selected node
      let context = "You're helping with a knowledge graph visualization. ";
      if (selectedNode) {
        context += `The user is looking at a node labeled "${selectedNode.label}" of type "${selectedNode.type}". `;
        
        if (selectedNode.entity_type) {
          context += `It's an entity of type "${selectedNode.entity_type}". `;
        }
        
        if (selectedNode.terms && selectedNode.terms.length > 0) {
          context += `Associated terms: ${selectedNode.terms.join(', ')}. `;
        }
      }
      
      // Add info about connected nodes if available
      if (selectedNode && graphData) {
        const connectedNodes = getConnectedNodes(selectedNode.id, graphData);
        if (connectedNodes.length > 0) {
          context += `Connected to nodes: ${connectedNodes.map(n => n.label).join(', ')}. `;
        }
      }
      //`Bearer ${process.env.REACT_APP_OPENROUTER_API_KEY}`
      // OpenRouter API integration - added loading delay to prevent UI freezing
      setTimeout(async () => {
        try {
          console.log(context)
          const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization':`Bearer ${api_key}`,
              'HTTP-Referer': window.location.origin,
              'X-Title': 'Knowledge Graph Assistant'
            },
            body: JSON.stringify({
              model: 'mistralai/mistral-small-3.1-24b-instruct:free',  // You can change the model as needed
              messages: [
                { role: 'system', content: context + 'Answer questions about the knowledge graph and explain concepts. Keep answers concise and helpful while explaining like the best teacher in world.' },
                { role: 'user', content: inputValue }
              ]
            })
          });

          const data = await response.json();
          
          // Add AI response to chat
          const aiMessage = {
            sender: 'ai',
            text: data.choices[0].message.content,
            timestamp: new Date().toISOString()
          };
          
          setMessages(prev => [...prev, aiMessage]);
        } catch (error) {
          console.error('Error sending message to OpenRouter:', error);
          
          // Add error message to chat
          const errorMessage = {
            sender: 'system',
            text: 'Sorry, there was an error processing your request. Please try again.',
            timestamp: new Date().toISOString()
          };
          
          setMessages(prev => [...prev, errorMessage]);
        } finally {
          setIsLoading(false);
        }
      }, 100); // Small delay to prevent UI freezing
    } catch (error) {
      console.error('Error in message processing:', error);
      setIsLoading(false);
    }
  };

  // Handle input change
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  // Handle pressing Enter key
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div 
      className={`chat-widget ${isOpen ? 'open' : 'closed'}`}
      ref={chatContainerRef}
    >
      <div className="chat-header" onClick={() => setIsOpen(!isOpen)}>
        <span>{isOpen ? 'Close Chat' : 'Chat with Knowledge Graph'}</span>
        <button>{isOpen ? '✕' : '?'}</button>
      </div>
      
      {isOpen && (
        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Ask questions about the knowledge graph or selected node.</p>
                <p>Examples:</p>
                <ul>
                  <li>What does this node represent?</li>
                  <li>How are these concepts connected?</li>
                  <li>Can you explain this topic in more detail?</li>
                </ul>
              </div>
            ) : (
              messages.map((msg, index) => (
                <div key={index} className={`message ${msg.sender}`}>
                  <div className="message-content">{msg.text}</div>
                  <div className="message-timestamp">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message ai loading">
                <div className="loading-indicator">
                  <span></span><span></span><span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <div className="chat-input">
            <textarea
              value={inputValue}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              placeholder="Type your question..."
              disabled={isLoading}
            />
            <button onClick={sendMessage} disabled={isLoading || !inputValue.trim()}>
              {isLoading ? '...' : 'Send'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

function KnowledgeGraph() {
  const { fileName } = useParams();
  const containerRef = useRef();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const graphRef = useRef(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [expandedNodes, setExpandedNodes] = useState(new Set(['doc_root']));
  
  // Animation states
  const [highlightedNodes, setHighlightedNodes] = useState(new Set());
  const [sidebarVisible, setSidebarVisible] = useState(false);
  
  const [pdfViewerOpen, setPdfViewerOpen] = useState(false);
  const navigate = useNavigate();
  
  // States for search and filtering
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedNodeTypes, setSelectedNodeTypes] = useState(['DOCUMENT', 'MAIN_TOPIC', 'SUBTOPIC', 'CONCEPT', 'ENTITY']);
  const [originalGraphData, setOriginalGraphData] = useState(null);
  const [filteredGraphData, setFilteredGraphData] = useState(null);
  const [graphMetadata, setGraphMetadata] = useState(null);
  
  // States for selected node and connected topics
  const [selectedNode, setSelectedNode] = useState(null);
  const [connectedTopics, setConnectedTopics] = useState([]);

  // Node type options for the filter
  const nodeTypes = [
    { value: 'DOCUMENT', label: 'Document' },
    { value: 'MAIN_TOPIC', label: 'Main Topics' },
    { value: 'SUBTOPIC', label: 'Subtopics' },
    { value: 'CONCEPT', label: 'Concepts' },
    { value: 'ENTITY', label: 'Entities' }
  ];

  // Search and filter function
  const filterGraph = () => {
    if (!originalGraphData) return;

    const searchLower = searchTerm.toLowerCase();
    
    // Filter nodes based on search term and selected types
    const filteredNodes = originalGraphData.nodes.filter(node => {
      const matchesSearch = node.label?.toLowerCase().includes(searchLower) || searchTerm === '';
      const matchesType = selectedNodeTypes.includes(node.type);
      return matchesSearch && matchesType;
    });

    // Get IDs of filtered nodes
    const filteredNodeIds = new Set(filteredNodes.map(node => node.id));

    // Filter links to only include connections between filtered nodes
    const filteredLinks = originalGraphData.links.filter(link => {
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
      return filteredNodeIds.has(sourceId) && filteredNodeIds.has(targetId);
    });

    setFilteredGraphData({ nodes: filteredNodes, links: filteredLinks });
    
    if (graphRef.current) {
      // Highlight newly filtered nodes with animation
      const newNodes = new Set(filteredNodes.map(node => node.id));
      setHighlightedNodes(newNodes);
      
      graphRef.current.graphData({ nodes: filteredNodes, links: filteredLinks });
      
      // Apply visual highlight to nodes
      setTimeout(() => {
        setHighlightedNodes(new Set());
      }, 1000);
    }
  };
// Handle search input change
  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };
  const togglePdfViewer = () => {
    setPdfViewerOpen(!pdfViewerOpen);
  };
  // Handle node type filter change
  const handleNodeTypeChange = (type) => {
    setSelectedNodeTypes(prev => {
      if (prev.includes(type)) {
        return prev.filter(t => t !== type);
      } else {
        return [...prev, type];
      }
    });
  };

  // Navigate to selected topic
  const handleTopicClick = (topic) => {
    if (!graphRef.current || !topic) return;
    
    // Find the node in the graph
    const node = graphRef.current.graphData().nodes.find(n => n.id === topic.id);
    if (node) {
      // Set as selected node
      setSelectedNode(node);
      
      // Expand the node
      const newExpanded = new Set(expandedNodes);
      newExpanded.add(node.id);
      setExpandedNodes(newExpanded);
      
      // Update visible nodes
      updateVisibleNodes(filteredGraphData || originalGraphData, newExpanded);
      
      // Add highlight animation to the selected node
      setHighlightedNodes(new Set([node.id]));
      setTimeout(() => setHighlightedNodes(new Set()), 1000);
      
      // Center view on the node with smooth animation
      graphRef.current.cameraPosition(
        { x: node.x * 1.4, y: node.y * 1.4, z: node.z * 1.4 },
        node,
        2000
      );
    }
  };

  // Effect to apply filters when search term or node types change
  useEffect(() => {
    filterGraph();
  }, [searchTerm, selectedNodeTypes]);

  // Effect to update connected topics when selected node changes
  useEffect(() => {
    if (selectedNode && filteredGraphData) {
      const connected = getConnectedNodes(selectedNode.id, filteredGraphData);
      setConnectedTopics(connected);
      setSidebarVisible(true);
    } else {
      setConnectedTopics([]);
      setSidebarVisible(false);
    }
  }, [selectedNode, filteredGraphData]);

  // Effect to handle sidebar transitions
  useEffect(() => {
    const sidebarElement = document.querySelector('.node-details-sidebar');
    if (sidebarElement) {
      if (sidebarVisible) {
        sidebarElement.classList.remove('closed');
      } else {
        sidebarElement.classList.add('closed');
      }
    }
  }, [sidebarVisible]);

  useEffect(() => {
    let handleResize;
    let animationFrameId;

    const initGraph = async () => {
      console.log(fileName, "Loading graph for file");
      if (!fileName || !containerRef.current) return;

      try {
        setLoading(true);
        const response = await fetch(`http://localhost:5000/api/knowledge-graph?file_name=${fileName}`);
        console.log(response);
        if (!response.ok) throw new Error('Failed to fetch knowledge graph data');

        const data = await response.json();
        const graphData = data.graph;
        
        // Store metadata if available
        if (graphData.metadata) {
          setGraphMetadata(graphData.metadata);
        }

        // Process backend data to fit our visualization structure
        const processedData = processBackendData(graphData);
        setOriginalGraphData(processedData);
        setFilteredGraphData(processedData);

        // Initialize force graph with improved performance settings
        graphRef.current = ForceGraph3D()(containerRef.current)
          .graphData(processedData)
          .backgroundColor('#000000')
          .nodeLabel(node => getNodeTooltip(node))
          .nodeColor(node => getNodeColor(node.type))
          .nodeRelSize(8)
          .nodeVal(node => node.size || (node.importance ? node.importance * 2 : 10))
          .width(window.innerWidth)
          .height(window.innerHeight - 60)
          .linkWidth(link => link.value || link.weight || 1)
          .linkOpacity(0.5)
          .linkLabel(link => getLinkLabel(link))
          .nodeThreeObject(node => {
            const group = new THREE.Group();
            
            // Scale node size based on importance
            const baseRadius = Math.max(5, 10 + (node.importance || 1) * 1.5);
            
            // Create node sphere with better performance material
            const geometry = new THREE.SphereGeometry(baseRadius);
            const material = new THREE.MeshLambertMaterial({
              color: getNodeColor(node.type),
              transparent: true,
              opacity: 0.8,
              emissive: getNodeColor(node.type),
              emissiveIntensity: 0.2
            });
            
            const sphere = new THREE.Mesh(geometry, material);
            
            // Add highlight animation if node is in highlighted set
            if (highlightedNodes.has(node.id)) {
              sphere.scale.set(1.2, 1.2, 1.2);
            }
            
            group.add(sphere);

            // Add label if node is expanded or important
            if (expandedNodes.has(node.id) || node.type === 'DOCUMENT' || node.type === 'MAIN_TOPIC') {
              const sprite = new SpriteText(
                node.label,
                baseRadius * 1.5,
                '#ffffff',
                'rgba(0,0,0,0.8)'
              );
              sprite.position.y = baseRadius + 2;
              sprite.fontWeight = node.type === 'DOCUMENT' || node.type === 'MAIN_TOPIC' ? 'bold' : 'normal';
              group.add(sprite);
            }

            return group;
          })
          .onNodeClick(node => {
            // Set selected node for connected topics display
            setSelectedNode(node);
            
            // Toggle node expansion
            const newExpanded = new Set(expandedNodes);
            if (newExpanded.has(node.id)) {
              // Collapse node and its children
              const toRemove = getDescendants(node, processedData);
              toRemove.forEach(id => newExpanded.delete(id));
            } else {
              // Expand node
              newExpanded.add(node.id);
            }
            setExpandedNodes(newExpanded);
            
            // Update visible nodes and links
            updateVisibleNodes(processedData, newExpanded);
            
            // Add highlight animation
            setHighlightedNodes(new Set([node.id]));
            setTimeout(() => setHighlightedNodes(new Set()), 1000);
            
            // Center view on clicked node with smooth animation
            graphRef.current.cameraPosition(
              { x: node.x * 1.4, y: node.y * 1.4, z: node.z * 1.4 },
              node,
              2000
            );
          });

        // Configure forces for hierarchical layout
        graphRef.current
          .d3Force('link', forceLink()
            .id(d => d.id)
            .distance(link => {
              // Adjust link distance based on node types
              const sourceType = link.source.type || '';
              const targetType = link.target.type || '';
              
              if (sourceType === 'DOCUMENT') return 150;
              if (sourceType === 'MAIN_TOPIC' && targetType === 'SUBTOPIC') return 100;
              if (targetType === 'ENTITY' || targetType === 'CONCEPT') return 80;
              return 120;
            }))
          .d3Force('charge', forceManyBody().strength(-400))
          .d3Force('center', forceCenter())
          .d3Force('collision', forceCollide(node => 20 + (node.importance || 1) * 2));

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        graphRef.current.scene().add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(200, 200, 200);
        graphRef.current.scene().add(directionalLight);

        // Performance optimization - limit graph updates for better performance
        const animate = () => {
          if (graphRef.current && graphRef.current.d3Force) {
              graphRef.current.d3Force("tick");
          }
          animationFrameId = requestAnimationFrame(animate);
      };
      
        
        // Start animation loop
        animate();

        setLoading(false);

      } catch (err) {
        console.error('Error initializing graph:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    initGraph();

    // Handle window resize
    handleResize = () => {
      if (graphRef.current) {
        graphRef.current.width(window.innerWidth);
        graphRef.current.height(window.innerHeight - 60);
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      if (graphRef.current) {
        graphRef.current._destructor();
      }
      if (handleResize) {
        window.removeEventListener('resize', handleResize);
      }
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [fileName]);

  // Helper function to process backend data
  const processBackendData = (backendData) => {
    const nodes = backendData.nodes.map(node => ({
      ...node,
      // Generate initial positions for smoother animation
      x: Math.random() * 200 - 100,
      y: Math.random() * 200 - 100,
      z: Math.random() * 200 - 100,
      // Calculate node level for hierarchical display
      level: getNodeLevel(node.type),
      // Set size based on importance
      size: node.importance ? node.importance * 2 : 10
    }));

    // Process links with proper IDs
    const links = backendData.edges.map(edge => ({
      source: edge.source,
      target: edge.target,
      relationship: edge.relationship,
      label: edge.relationship,
      value: edge.weight || 1
    }));

    return { nodes, links };
  };

  const getDescendants = (node, data) => {
    const descendants = new Set();
    const traverse = (nodeId) => {
      descendants.add(nodeId);
      // Find all links where this node is the source
      const childLinks = data.links.filter(link => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        return sourceId === nodeId;
      });
      
      // Add all targets as descendants
      childLinks.forEach(link => {
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;
        if (!descendants.has(targetId)) {
          traverse(targetId);
        }
      });
    };
    
    traverse(node.id);
    return descendants;
  };

  const updateVisibleNodes = (data, expanded) => {
    try {
      // Find all nodes that should be visible
      // This includes:
      // 1. All expanded nodes
      // 2. Direct children of expanded nodes
      
      const visibleNodeIds = new Set();
      
      // Add all expanded nodes
      expanded.forEach(id => visibleNodeIds.add(id));
      
      // Add direct children of expanded nodes
      data.links.forEach(link => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;
        
        if (expanded.has(sourceId)) {
          visibleNodeIds.add(targetId);
        }
      });
      
      // Filter nodes and links
      const visibleNodes = data.nodes.filter(node => visibleNodeIds.has(node.id));
      const visibleLinks = data.links.filter(link => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;
        return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
      });

      if (graphRef.current) {
        // Update with animation for smooth transition
        graphRef.current.graphData({
          nodes: visibleNodes,
          links: visibleLinks
        });
      }
    } catch (error) {
      console.error('Error updating visible nodes:', error);
    }
  };

  const getNodeTooltip = (node) => {
    let tooltip = `${node.label} (${node.type})`;
    
    if (node.terms && node.terms.length > 0) {
      tooltip += `\nKey terms: ${node.terms.join(', ')}`;
    }
    
    if (node.entity_type) {
      tooltip += `\nEntity type: ${node.entity_type}`;
    }
    
    if (node.importance) {
      tooltip += `\nImportance: ${node.importance.toFixed(2)}`;
    }
    
    return tooltip;
  };

  const getLinkLabel = (link) => {
    return link.relationship || 'connected to';
  };

  const getNodeLevel = (type) => {
    switch (type) {
      case 'DOCUMENT':
        return 0;
      case 'MAIN_TOPIC':
        return 1;
      case 'SUBTOPIC':
        return 2;
      case 'CONCEPT':
        return 3;
      case 'ENTITY':
        return 4;
      default:
        return 3;
    }
  };

  const downloadCSV = () => { 
    if (!graphRef.current) return;
    
    const graphData = graphRef.current.graphData();
    console.log('Downloading graph data:', graphData);
    
    const sanitizeText = (text) => {
      if (!text) return 'null'; // Handle empty values
      text = String(text).replace(/\u0000/g, ''); // Remove null characters
      return `"${text.replace(/"/g, '""')}"`;
    };
  
    const nodesCSV = [
      ['id', 'label', 'type', 'importance', 'entity_type', 'terms'].join(','), // Header
      ...graphData.nodes.map(node => [
        sanitizeText(node.id),
        sanitizeText(node.label),
        sanitizeText(node.type),
        node.importance || '',
        sanitizeText(node.entity_type || ''),
        node.terms ? sanitizeText(node.terms.join(';')) : ''
      ].join(','))
    ].join('\n');
  
    // Create edges CSV
    const edgesCSV = [
      ['source', 'target', 'relationship', 'weight'].join(','), // Header
      ...graphData.links.map(link => [
        sanitizeText(typeof link.source === 'object' ? link.source.id : link.source),
        sanitizeText(typeof link.target === 'object' ? link.target.id : link.target),
        sanitizeText(link.relationship || ''),
        link.value || link.weight || 1
      ].join(','))
    ].join('\n');
    
    // Download function
    const downloadFile = (content, fileName) => {
      const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    };

    // Ensure fileName is provided
    const baseFileName = fileName ? fileName.replace('.pdf', '') : 'graph_data';

    // Download both files
    downloadFile(nodesCSV, `${baseFileName}_nodes.csv`);
    downloadFile(edgesCSV, `${baseFileName}_edges.csv`);
  };

  const closeGraph = () => {
    navigate('/');
  };

// Function to handle zoom controls
const handleZoom = (factor) => {
  if (!graphRef.current) return;
  
  const distance = graphRef.current.cameraPosition().z;
  const newDistance = distance / factor;
  
  graphRef.current.cameraPosition(
    { z: newDistance },
    // Look at the same point
    graphRef.current.cameraPosition().lookat,
    // Animation duration in ms
    800
  );
  
  setZoomLevel(factor > 1 ? zoomLevel + 0.1 : zoomLevel - 0.1);
};

// Function to reset the graph view
const resetView = () => {
  if (!graphRef.current) return;
  
  graphRef.current.cameraPosition(
    { x: 0, y: 0, z: 300 },
    { x: 0, y: 0, z: 0 },
    2000
  );
  
  setZoomLevel(1);
};

// Function to toggle fullscreen mode
const toggleFullscreen = () => {
  const container = containerRef.current;
  
  if (!document.fullscreenElement) {
    if (container.requestFullscreen) {
      container.requestFullscreen();
    } else if (container.mozRequestFullScreen) {
      container.mozRequestFullScreen();
    } else if (container.webkitRequestFullscreen) {
      container.webkitRequestFullscreen();
    } else if (container.msRequestFullscreen) {
      container.msRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.mozCancelFullScreen) {
      document.mozCancelFullScreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    } else if (document.msExitFullscreen) {
      document.msExitFullscreen();
    }
  }
};

// Function to export the graph as PNG
const exportPNG = () => {
  if (!graphRef.current) return;
  
  // Create a screenshot of the graph
  const renderer = graphRef.current.renderer();
  renderer.domElement.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${fileName ? fileName.replace(/\.[^/.]+$/, '') : 'knowledge-graph'}_visualization.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  });
};

return (
  <div className="knowledge-graph-container">
    <div className="graph-header">
      <h3>Knowledge Graph: {fileName}</h3>
      
      {graphMetadata && (
        <div className="graph-metadata">
          <span>Topics: {graphMetadata.topic_count}</span>
          <span>Subtopics: {graphMetadata.subtopic_count}</span>
          <span>Entities: {graphMetadata.entity_count}</span>
          <span>Concepts: {graphMetadata.concept_count}</span>
        </div>
      )}
      
      <div className="search-filter-container">
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={handleSearchChange}
            className="search-input"
          />
        </div>
        <div className="filter-options">
          {nodeTypes.map(type => (
            <label key={type.value} className="filter-checkbox">
              <input
                type="checkbox"
                checked={selectedNodeTypes.includes(type.value)}
                onChange={() => handleNodeTypeChange(type.value)}
              />
              {type.label}
            </label>
          ))}
        </div>
      </div>
      <div className="graph-controls">
        <button onClick={() => handleZoom(1.2)} className="zoom-btn" title="Zoom In">+</button>
        <button onClick={() => handleZoom(0.8)} className="zoom-btn" title="Zoom Out">-</button>
        <button onClick={resetView} className="reset-btn" title="Reset View">Reset</button>
        <button onClick={toggleFullscreen} className="fullscreen-btn" title="Toggle Fullscreen">⛶</button>
        <button onClick={exportPNG} className="export-btn" title="Export as PNG">Export PNG</button>
        <button 
          onClick={downloadCSV}
          className="download-btn"
          disabled={loading || error}
          title="Download Data as CSV"
        >
          Download CSV
        </button>
        <button onClick={togglePdfViewer} className="pdf-viewer-btn" title="View & Highlight PDF">View PDF</button>
        <button onClick={closeGraph} className="close-btn">Close</button>
      </div>
    </div>
    <div className="graph-content">
      {loading && (
        <div className="graph-loading">
          <div className="spinner"></div>
          <span>Loading graph...</span>
        </div>
      )}
      {error && <div className="graph-error">Error: {error}</div>}
      
      <div ref={containerRef} className="graph-container" />
      
      {/* Node state visualization legend */}
      <NodeStateVisualization />
      
      {/* Connected topics sidebar */}
      {selectedNode && (
        <div className={`node-details-sidebar ${!sidebarVisible ? 'closed' : ''}`}>
          <div className="sidebar-header">
            <h3>Selected Node: {selectedNode.label}</h3>
            <button 
              className="close-sidebar-btn" 
              onClick={() => setSidebarVisible(false)}
              title="Close sidebar"
            >
              ✕
            </button>
          </div>
          
          <div className="node-details">
            <p><strong>Type:</strong> {selectedNode.type}</p>
            {selectedNode.entity_type && <p><strong>Entity Type:</strong> {selectedNode.entity_type}</p>}
            {selectedNode.importance && <p><strong>Importance:</strong> {selectedNode.importance.toFixed(2)}</p>}
            
            {selectedNode.terms && selectedNode.terms.length > 0 && (
              <div className="terms-section">
                <p><strong>Associated Terms:</strong></p>
                <ul className="terms-list">
                  {selectedNode.terms.map((term, index) => (
                    <li key={index}>{term}</li>
                  ))}
                </ul>
              </div>
            )}
            
            <ConnectedTopicsList 
              topics={connectedTopics} 
              onTopicClick={handleTopicClick} 
            />
          </div>
        </div>
      )}
      
      {/* Chat widget */}
      <ChatWidget 
        selectedNode={selectedNode}
        graphData={filteredGraphData || originalGraphData}
      />
      
      <PDFViewer isOpen={pdfViewerOpen} onClose={() => setPdfViewerOpen(false)} 
            graphData={filteredGraphData || originalGraphData}
            selectedNodeTypes={selectedNodeTypes}/>
    </div>
  </div>
);
}

export default KnowledgeGraph;
// import React, { useEffect, useRef, useState } from 'react';
// import ForceGraph3D from '3d-force-graph';
// import * as THREE from 'three';
// import { 
//   forceSimulation,
//   forceManyBody,
//   forceLink,
//   forceCenter,
//   forceY,
//   forceCollide
// } from 'd3-force-3d';
// import '../styles/KnowledgeGraph.css';
// import { useNavigate, useParams } from 'react-router-dom';

// // Custom SpriteText class
// class SpriteText extends THREE.Sprite {
//   constructor(text, textHeight = 8, color = '#ffffff', backgroundColor = 'rgba(0,0,0,0.8)') {
//     super(new THREE.SpriteMaterial({ map: new THREE.Texture() }));
//     this.text = text;
//     this.textHeight = textHeight;
//     this.color = color;
//     this.backgroundColor = backgroundColor;
//     this.padding = 2;
//     this.updateTexture();
//   }

//   updateTexture() {
//     const canvas = document.createElement('canvas');
//     const ctx = canvas.getContext('2d');
//     ctx.font = `${this.textHeight}px Arial`;
    
//     const textWidth = ctx.measureText(this.text).width;
//     canvas.width = textWidth + (this.padding * 2);
//     canvas.height = this.textHeight + (this.padding * 2);
    
//     // Draw background
//     ctx.fillStyle = this.backgroundColor;
//     ctx.fillRect(0, 0, canvas.width, canvas.height);
    
//     // Draw text
//     ctx.font = `${this.textHeight}px Arial`;
//     ctx.fillStyle = this.color;
//     ctx.textBaseline = 'middle';
//     ctx.textAlign = 'center';
//     ctx.fillText(this.text, canvas.width / 2, canvas.height / 2);
    
//     const texture = new THREE.Texture(canvas);
//     texture.needsUpdate = true;
//     this.material.map = texture;
//     this.scale.set(canvas.width / canvas.height * this.textHeight, this.textHeight, 1);
//   }
// }

// function KnowledgeGraph() {
//   const { fileName } = useParams();
//   const containerRef = useRef();
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);
//   const graphRef = useRef(null);
//   const [zoomLevel, setZoomLevel] = useState(1);
//   const [expandedNodes, setExpandedNodes] = useState(new Set(['doc_root'])); // Updated to match backend root ID
  
//   const navigate = useNavigate();
  
//   // New states for search and filtering
//   const [searchTerm, setSearchTerm] = useState('');
//   const [selectedNodeTypes, setSelectedNodeTypes] = useState(['DOCUMENT', 'MAIN_TOPIC', 'SUBTOPIC', 'CONCEPT', 'ENTITY']);
//   const [originalGraphData, setOriginalGraphData] = useState(null);
//   const [filteredGraphData, setFilteredGraphData] = useState(null);
//   const [graphMetadata, setGraphMetadata] = useState(null);

//   // Node type options for the filter - updated to match backend types
//   const nodeTypes = [
//     { value: 'DOCUMENT', label: 'Document' },
//     { value: 'MAIN_TOPIC', label: 'Main Topics' },
//     { value: 'SUBTOPIC', label: 'Subtopics' },
//     { value: 'CONCEPT', label: 'Concepts' },
//     { value: 'ENTITY', label: 'Entities' }
//   ];

//   // Search and filter function
//   const filterGraph = () => {
//     if (!originalGraphData) return;

//     const searchLower = searchTerm.toLowerCase();
    
//     // Filter nodes based on search term and selected types
//     const filteredNodes = originalGraphData.nodes.filter(node => {
//       const matchesSearch = node.label?.toLowerCase().includes(searchLower) || searchTerm === '';
//       const matchesType = selectedNodeTypes.includes(node.type);
//       return matchesSearch && matchesType;
//     });

//     // Get IDs of filtered nodes
//     const filteredNodeIds = new Set(filteredNodes.map(node => node.id));

//     // Filter links to only include connections between filtered nodes
//     const filteredLinks = originalGraphData.links.filter(link => {
//       const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
//       const targetId = typeof link.target === 'object' ? link.target.id : link.target;
//       return filteredNodeIds.has(sourceId) && filteredNodeIds.has(targetId);
//     });

//     setFilteredGraphData({ nodes: filteredNodes, links: filteredLinks });
    
//     if (graphRef.current) {
//       graphRef.current.graphData({ nodes: filteredNodes, links: filteredLinks });
//     }
//   };

//   // Handle search input change
//   const handleSearchChange = (e) => {
//     setSearchTerm(e.target.value);
//   };

//   // Handle node type filter change
//   const handleNodeTypeChange = (type) => {
//     setSelectedNodeTypes(prev => {
//       if (prev.includes(type)) {
//         return prev.filter(t => t !== type);
//       } else {
//         return [...prev, type];
//       }
//     });
//   };

//   // Effect to apply filters when search term or node types change
//   useEffect(() => {
//     filterGraph();
//   }, [searchTerm, selectedNodeTypes]);

//   useEffect(() => {
//     let handleResize;

//     const initGraph = async () => {
//       console.log(fileName, "Loading graph for file");
//       if (!fileName || !containerRef.current) return;

//       try {
//         setLoading(true);
//         const response = await fetch(`http://localhost:5000/api/knowledge-graph?file_name=${fileName}`);
//         console.log(response);
//         if (!response.ok) throw new Error('Failed to fetch knowledge graph data');

//         const data = await response.json();
//         const graphData = data.graph;
        
//         // Store metadata if available
//         if (graphData.metadata) {
//           setGraphMetadata(graphData.metadata);
//         }

//         // Process backend data to fit our visualization structure
//         const processedData = processBackendData(graphData);
//         setOriginalGraphData(processedData);
//         setFilteredGraphData(processedData);

//         // Initialize force graph
//         graphRef.current = ForceGraph3D()(containerRef.current)
//           .graphData(processedData)
//           .backgroundColor('#000000')
//           .nodeLabel(node => getNodeTooltip(node))
//           .nodeColor(node => getNodeColor(node.type))
//           .nodeRelSize(8)
//           .nodeVal(node => node.size || (node.importance ? node.importance * 2 : 10))
//           .width(window.innerWidth)
//           .height(window.innerHeight - 60)
//           .linkWidth(link => link.value || link.weight || 1)
//           .linkOpacity(0.5)
//           .linkLabel(link => getLinkLabel(link))
//           .nodeThreeObject(node => {
//             const group = new THREE.Group();
            
//             // Scale node size based on importance
//             const baseRadius = Math.max(5, 10 + (node.importance || 1) * 1.5);
            
//             // Create node sphere
//             const geometry = new THREE.SphereGeometry(baseRadius);
//             const material = new THREE.MeshPhongMaterial({
//               color: getNodeColor(node.type),
//               transparent: true,
//               opacity: 0.8,
//               shininess: 100,
//               emissive: getNodeColor(node.type),
//               emissiveIntensity: 0.2
//             });
//             const sphere = new THREE.Mesh(geometry, material);
//             group.add(sphere);

//             // Add label if node is expanded or important
//             if (expandedNodes.has(node.id) || node.type === 'DOCUMENT' || node.type === 'MAIN_TOPIC') {
//               const sprite = new SpriteText(
//                 node.label,
//                 baseRadius * 1.5,
//                 '#ffffff',
//                 'rgba(0,0,0,0.8)'
//               );
//               sprite.position.y = baseRadius + 2;
//               sprite.fontWeight = node.type === 'DOCUMENT' || node.type === 'MAIN_TOPIC' ? 'bold' : 'normal';
//               group.add(sprite);
//             }

//             return group;
//           })
//           .onNodeClick(node => {
//             // Toggle node expansion
//             const newExpanded = new Set(expandedNodes);
//             if (newExpanded.has(node.id)) {
//               // Collapse node and its children
//               const toRemove = getDescendants(node, processedData);
//               toRemove.forEach(id => newExpanded.delete(id));
//             } else {
//               // Expand node
//               newExpanded.add(node.id);
//             }
//             setExpandedNodes(newExpanded);
            
//             // Update visible nodes and links
//             updateVisibleNodes(processedData, newExpanded);
            
//             // Center view on clicked node
//             graphRef.current.cameraPosition(
//               { x: node.x * 1.4, y: node.y * 1.4, z: node.z * 1.4 },
//               node,
//               2000
//             );
//           });

//         // Configure forces for hierarchical layout
//         graphRef.current
//           .d3Force('link', forceLink()
//             .id(d => d.id)
//             .distance(link => {
//               // Adjust link distance based on node types
//               const sourceType = link.source.type || '';
//               const targetType = link.target.type || '';
              
//               if (sourceType === 'DOCUMENT') return 150;
//               if (sourceType === 'MAIN_TOPIC' && targetType === 'SUBTOPIC') return 100;
//               if (targetType === 'ENTITY' || targetType === 'CONCEPT') return 80;
//               return 120;
//             }))
//           .d3Force('charge', forceManyBody().strength(-400))
//           .d3Force('center', forceCenter())
//           .d3Force('collision', forceCollide(node => 20 + (node.importance || 1) * 2));

//         // Add lights
//         const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
//         graphRef.current.scene().add(ambientLight);

//         const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
//         directionalLight.position.set(200, 200, 200);
//         graphRef.current.scene().add(directionalLight);

//         setLoading(false);

//       } catch (err) {
//         console.error('Error initializing graph:', err);
//         setError(err.message);
//         setLoading(false);
//       }
//     };

//     initGraph();

//     // Handle window resize
//     handleResize = () => {
//       if (graphRef.current) {
//         graphRef.current.width(window.innerWidth);
//         graphRef.current.height(window.innerHeight - 60);
//       }
//     };
//     window.addEventListener('resize', handleResize);

//     return () => {
//       if (graphRef.current) {
//         graphRef.current._destructor();
//       }
//       if (handleResize) {
//         window.removeEventListener('resize', handleResize);
//       }
//     };
//   }, [fileName]);

//   // Helper function to process backend data
//   const processBackendData = (backendData) => {
//     const nodes = backendData.nodes.map(node => ({
//       ...node,
//       // Generate initial positions for smoother animation
//       x: Math.random() * 200 - 100,
//       y: Math.random() * 200 - 100,
//       z: Math.random() * 200 - 100,
//       // Calculate node level for hierarchical display
//       level: getNodeLevel(node.type),
//       // Set size based on importance
//       size: node.importance ? node.importance * 2 : 10
//     }));

//     // Process links with proper IDs
//     const links = backendData.edges.map(edge => ({
//       source: edge.source,
//       target: edge.target,
//       relationship: edge.relationship,
//       label: edge.relationship,
//       value: edge.weight || 1
//     }));

//     return { nodes, links };
//   };

//   const getDescendants = (node, data) => {
//     const descendants = new Set();
//     const traverse = (nodeId) => {
//       descendants.add(nodeId);
//       // Find all links where this node is the source
//       const childLinks = data.links.filter(link => {
//         const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
//         return sourceId === nodeId;
//       });
      
//       // Add all targets as descendants
//       childLinks.forEach(link => {
//         const targetId = typeof link.target === 'object' ? link.target.id : link.target;
//         if (!descendants.has(targetId)) {
//           traverse(targetId);
//         }
//       });
//     };
    
//     traverse(node.id);
//     return descendants;
//   };

//   const updateVisibleNodes = (data, expanded) => {
//     try {
//       // Find all nodes that should be visible
//       // This includes:
//       // 1. All expanded nodes
//       // 2. Direct children of expanded nodes
      
//       const visibleNodeIds = new Set();
      
//       // Add all expanded nodes
//       expanded.forEach(id => visibleNodeIds.add(id));
      
//       // Add direct children of expanded nodes
//       data.links.forEach(link => {
//         const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
//         const targetId = typeof link.target === 'object' ? link.target.id : link.target;
        
//         if (expanded.has(sourceId)) {
//           visibleNodeIds.add(targetId);
//         }
//       });
      
//       // Filter nodes and links
//       const visibleNodes = data.nodes.filter(node => visibleNodeIds.has(node.id));
//       const visibleLinks = data.links.filter(link => {
//         const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
//         const targetId = typeof link.target === 'object' ? link.target.id : link.target;
//         return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
//       });

//       if (graphRef.current) {
//         graphRef.current.graphData({
//           nodes: visibleNodes,
//           links: visibleLinks
//         });
//       }
//     } catch (error) {
//       console.error('Error updating visible nodes:', error);
//     }
//   };

//   // Updated color scheme for better visibility - mapped to backend node types
//   const getNodeColor = (type) => {
//     switch (type) {
//       case 'DOCUMENT':
//         return '#FFFFFF'; // White for document root
//       case 'MAIN_TOPIC':
//         return '#4CAF50'; // Bright green
//       case 'SUBTOPIC':
//         return '#2196F3'; // Bright blue
//       case 'CONCEPT':
//         return '#FFC107'; // Bright yellow
//       case 'ENTITY':
//         return '#E91E63'; // Bright pink
//       default:
//         return '#FF5722'; // Bright orange
//     }
//   };

//   const getNodeTooltip = (node) => {
//     let tooltip = `${node.label} (${node.type})`;
    
//     if (node.terms && node.terms.length > 0) {
//       tooltip += `\nKey terms: ${node.terms.join(', ')}`;
//     }
    
//     if (node.entity_type) {
//       tooltip += `\nEntity type: ${node.entity_type}`;
//     }
    
//     if (node.importance) {
//       tooltip += `\nImportance: ${node.importance.toFixed(2)}`;
//     }
    
//     return tooltip;
//   };

//   const getLinkLabel = (link) => {
//     return link.relationship || 'connected to';
//   };

//   const getNodeLevel = (type) => {
//     switch (type) {
//       case 'DOCUMENT':
//         return 0;
//       case 'MAIN_TOPIC':
//         return 1;
//       case 'SUBTOPIC':
//         return 2;
//       case 'CONCEPT':
//         return 3;
//       case 'ENTITY':
//         return 4;
//       default:
//         return 3;
//     }
//   };

//   const downloadCSV = () => { 
//     if (!graphRef.current) return;
    
//     const graphData = graphRef.current.graphData();
//     console.log('Downloading graph data:', graphData);
    
//     const sanitizeText = (text) => {
//       if (!text) return 'null'; // Handle empty values
//       text = String(text).replace(/\u0000/g, ''); // Remove null characters
//       return `"${text.replace(/"/g, '""')}"`;
//     };
  
//     const nodesCSV = [
//       ['id', 'label', 'type', 'importance', 'entity_type', 'terms'].join(','), // Header
//       ...graphData.nodes.map(node => [
//         sanitizeText(node.id),
//         sanitizeText(node.label),
//         sanitizeText(node.type),
//         node.importance || '',
//         sanitizeText(node.entity_type || ''),
//         node.terms ? sanitizeText(node.terms.join(';')) : ''
//       ].join(','))
//     ].join('\n');
  
//     // Create edges CSV
//     const edgesCSV = [
//       ['source', 'target', 'relationship', 'weight'].join(','), // Header
//       ...graphData.links.map(link => [
//         sanitizeText(typeof link.source === 'object' ? link.source.id : link.source),
//         sanitizeText(typeof link.target === 'object' ? link.target.id : link.target),
//         sanitizeText(link.relationship || ''),
//         link.value || link.weight || 1
//       ].join(','))
//     ].join('\n');
    
//     // Download function
//     const downloadFile = (content, fileName) => {
//       const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
//       const link = document.createElement('a');
//       const url = URL.createObjectURL(blob);
//       link.href = url;
//       link.download = fileName;
//       document.body.appendChild(link);
//       link.click();
//       document.body.removeChild(link);
//       URL.revokeObjectURL(url);
//     };

//     // Ensure fileName is provided
//     const baseFileName = fileName ? fileName.replace('.pdf', '') : 'graph_data';

//     // Download both files
//     downloadFile(nodesCSV, `${baseFileName}_nodes.csv`);
//     downloadFile(edgesCSV, `${baseFileName}_edges.csv`);
//   };

//   const closeGraph = () => {
//     navigate('/');
//   };

//   return (
//     <div className="knowledge-graph-container">
//       <div className="graph-header">
//         <h3>Knowledge Graph: {fileName}</h3>
        
//         {graphMetadata && (
//           <div className="graph-metadata">
//             <span>Topics: {graphMetadata.topic_count}</span>
//             <span>Subtopics: {graphMetadata.subtopic_count}</span>
//             <span>Entities: {graphMetadata.entity_count}</span>
//             <span>Concepts: {graphMetadata.concept_count}</span>
//           </div>
//         )}
        
//         <div className="search-filter-container">
//           <div className="search-bar">
//             <input
//               type="text"
//               placeholder="Search nodes..."
//               value={searchTerm}
//               onChange={handleSearchChange}
//               className="search-input"
//             />
//           </div>
//           <div className="filter-options">
//             {nodeTypes.map(type => (
//               <label key={type.value} className="filter-checkbox">
//                 <input
//                   type="checkbox"
//                   checked={selectedNodeTypes.includes(type.value)}
//                   onChange={() => handleNodeTypeChange(type.value)}
//                 />
//                 {type.label}
//               </label>
//             ))}
//           </div>
//         </div>
//         <div className="graph-controls">
//           <button 
//             onClick={downloadCSV}
//             className="download-btn"
//             disabled={loading || error}
//           >
//             Download CSV
//           </button>
//           <button onClick={closeGraph}>Close</button>
//         </div>
//       </div>
//       <div className="graph-content">
//         {loading && <div className="graph-loading">Loading graph...</div>}
//         {error && <div className="graph-error">Error: {error}</div>}
//         <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
//       </div>
//     </div>
//   );
// }

// export default KnowledgeGraph;




// import React, { useEffect, useRef, useState } from 'react';
// import ForceGraph3D from '3d-force-graph';
// import * as THREE from 'three';
// import { 
//   forceSimulation,
//   forceManyBody,
//   forceLink,
//   forceCenter,
//   forceY,
//   forceCollide
// } from 'd3-force-3d';
// import '../styles/KnowledgeGraph.css';
// import { useNavigate, useParams } from 'react-router-dom';
// // // Custom SpriteText class
// class SpriteText extends THREE.Sprite {
//   constructor(text, textHeight = 8, color = '#ffffff', backgroundColor = 'rgba(0,0,0,0.8)') {
//     super(new THREE.SpriteMaterial({ map: new THREE.Texture() }));
//     this.text = text;
//     this.textHeight = textHeight;
//     this.color = color;
//     this.backgroundColor = backgroundColor;
//     this.padding = 2;
//     this.updateTexture();
//   }

//   updateTexture() {
//     const canvas = document.createElement('canvas');
//     const ctx = canvas.getContext('2d');
//     ctx.font = `${this.textHeight}px Arial`;
    

//     const textWidth = ctx.measureText(this.text).width;
//     canvas.width = textWidth + (this.padding * 2);
//     canvas.height = this.textHeight + (this.padding * 2);
    
//     // Draw background
//     ctx.fillStyle = this.backgroundColor;
//     ctx.fillRect(0, 0, canvas.width, canvas.height);
    
//     // Draw text
//     ctx.font = `${this.textHeight}px Arial`;
//     ctx.fillStyle = this.color;
//     ctx.textBaseline = 'middle';
//     ctx.textAlign = 'center';
//     ctx.fillText(this.text, canvas.width / 2, canvas.height / 2);
    

//     const texture = new THREE.Texture(canvas);
//     texture.needsUpdate = true;
//     this.material.map = texture;
//     this.scale.set(canvas.width / canvas.height * this.textHeight, this.textHeight, 1);
//   }
// }

// function KnowledgeGraph() {
//   const { fileName } = useParams();
//   const containerRef = useRef();
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);
//   const graphRef = useRef(null);
//   const [zoomLevel, setZoomLevel] = useState(1);
//   const [expandedNodes, setExpandedNodes] = useState(new Set(['root']));
  
//   const navigate=useNavigate();
  
//   // New states for search and filtering
//   const [searchTerm, setSearchTerm] = useState('');
//   const [selectedNodeTypes, setSelectedNodeTypes] = useState(['MAIN_TOPIC', 'SUBTOPIC', 'CONCEPT', 'ENTITY', 'METADATA']);
//   const [originalGraphData, setOriginalGraphData] = useState(null);
//   const [filteredGraphData, setFilteredGraphData] = useState(null);

//   // Node type options for the filter
//   const nodeTypes = [
//     { value: 'MAIN_TOPIC', label: 'Main Topics' },
//     { value: 'SUBTOPIC', label: 'Subtopics' },
//     { value: 'CONCEPT', label: 'Concepts' },
//     { value: 'ENTITY', label: 'Entities' },
//     { value: 'METADATA', label: 'Metadata' }
//   ];

//   // Search and filter function
//   const filterGraph = () => {
//     if (!originalGraphData) return;

//     const searchLower = searchTerm.toLowerCase();
    
//     // Filter nodes based on search term and selected types
//     const filteredNodes = originalGraphData.nodes.filter(node => {
//       const matchesSearch = node.label?.toLowerCase().includes(searchLower) || searchTerm === '';
//       const matchesType = selectedNodeTypes.includes(node.type) || node.type === 'ROOT';
//       return matchesSearch && matchesType;
//     });

//     // Get IDs of filtered nodes
//     const filteredNodeIds = new Set(filteredNodes.map(node => node.id));

//     // Filter links to only include connections between filtered nodes
//     const filteredLinks = originalGraphData.links.filter(link => {
//       const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
//       const targetId = typeof link.target === 'object' ? link.target.id : link.target;
//       return filteredNodeIds.has(sourceId) && filteredNodeIds.has(targetId);
//     });

//     setFilteredGraphData({ nodes: filteredNodes, links: filteredLinks });
    
//     if (graphRef.current) {
//       graphRef.current.graphData({ nodes: filteredNodes, links: filteredLinks });
//     }
//   };

//   // Handle search input change
//   const handleSearchChange = (e) => {
//     setSearchTerm(e.target.value);
//   };

//   // Handle node type filter change
//   const handleNodeTypeChange = (type) => {
//     setSelectedNodeTypes(prev => {
//       if (prev.includes(type)) {
//         return prev.filter(t => t !== type);
//       } else {
//         return [...prev, type];
//       }
//     });
//   };

//   // Effect to apply filters when search term or node types change
//   useEffect(() => {
//     filterGraph();
//   }, [searchTerm, selectedNodeTypes]);

//   useEffect(() => {
//     let handleResize;

//     const initGraph = async () => {
//       console.log(fileName,"adadasdsa")
//       if (!fileName || !containerRef.current) return;

//       try {
//         setLoading(true);
//         const response = await fetch(`http://localhost:5000/api/knowledge-graph?file_name=${fileName}`);
//         if (!response.ok) throw new Error('Failed to fetch knowledge graph data');

//         const data = await response.json();
//         const graphData = data.graph;

//         // Process data into hierarchical structure
//         const processedData = processHierarchicalData(graphData);
//         setOriginalGraphData(processedData);
//         setFilteredGraphData(processedData);

//         // Initialize force graph with radial layout
        
// //         // Initialize force graph with radial layout
//         graphRef.current = ForceGraph3D()(containerRef.current)
//           .graphData(processedData)
//           .backgroundColor('#000000')
//           .nodeLabel('label')
//           .nodeColor(node => node.color)
//           .nodeRelSize(8)
//           .nodeVal(node => node.size)
//           .width(window.innerWidth)
//           .height(window.innerHeight - 60)
//           .linkWidth(2)
//           .linkOpacity(0.5)
//           .nodeThreeObject(node => {
//             const group = new THREE.Group();
            
//             // Scale node size based on level and importance
//             const baseRadius = Math.max(5, 20 - node.level * 3);
//             const radius = baseRadius * (node.importance || 1);
            
//             // Create node sphere
//             const geometry = new THREE.SphereGeometry(radius);
//             const material = new THREE.MeshPhongMaterial({
//               color: node.color,
//               transparent: true,
//               opacity: 0.8,
//               shininess: 100,
//               emissive: node.color,
//               emissiveIntensity: 0.2
//             });
//             const sphere = new THREE.Mesh(geometry, material);
//             group.add(sphere);

//             // Add label if node is expanded or important
//             if (expandedNodes.has(node.id) || node.level <= 1) {
//               const sprite = new SpriteText(
//                 node.label,
//                 radius * 2,
//                 '#ffffff',
//                 'rgba(0,0,0,0.8)'
//               );
//               sprite.position.y = radius + 2;
//               sprite.fontWeight = 'bold';
//               group.add(sprite);
//             }

//             return group;
//           })
//           .onNodeClick(node => {
//             // Toggle node expansion
//             const newExpanded = new Set(expandedNodes);
//             if (newExpanded.has(node.id)) {
//               // Collapse node and its children
//               const toRemove = getDescendants(node, processedData);
//               toRemove.forEach(id => newExpanded.delete(id));
//             } else {
//               // Expand node
//               newExpanded.add(node.id);
//             }
//             setExpandedNodes(newExpanded);
            
//             // Update visible nodes and links
//             updateVisibleNodes(processedData, newExpanded);
            
//             // Center view on clicked node
//             graphRef.current.cameraPosition(
//               { x: node.x * 1.4, y: node.y * 1.4, z: node.z * 1.4 },
//               node,
//               2000
//             );
//           });

//         // Configure forces for radial layout
//         const radialForce = alpha => {
//           processedData.nodes.forEach(node => {
//             if (!node.x && !node.y) return; // Skip nodes without position

//             // Calculate ideal radial position based on level
//             const angle = (node.index || 0) * (2 * Math.PI / (node.siblings || 1));
//             const radius = node.level * 200; // Increase radius for each level
            
//             const targetX = Math.cos(angle) * radius;
//             const targetY = Math.sin(angle) * radius;
//             const targetZ = node.level * 50; // Slight vertical separation

//             // Apply force towards ideal position
//             node.vx += (targetX - node.x) * alpha * 0.1;
//             node.vy += (targetY - node.y) * alpha * 0.1;
//             node.vz += (targetZ - node.z) * alpha * 0.1;
//           });
//         };

//         graphRef.current
//           .d3Force('link', forceLink()
//             .id(d => d.id)
//             .distance(link => 100 + link.source.level * 50))
//           .d3Force('charge', forceManyBody().strength(-1000))
//           .d3Force('center', forceCenter())
//           .d3Force('radial', radialForce)
//           .d3Force('collision', forceCollide(node => 30 + node.level * 10));

//         // Add lights
//         const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
//         graphRef.current.scene().add(ambientLight);

//         const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
//         directionalLight.position.set(200, 200, 200);
//         graphRef.current.scene().add(directionalLight);

//         setLoading(false);

//       } catch (err) {
//         console.error('Error initializing graph:', err);
//         setError(err.message);
//         setLoading(false);
//       }
//     };

//     initGraph();

//     return () => {
//       if (graphRef.current) {
//         graphRef.current._destructor();
//       }
//       if (handleResize) {
//         window.removeEventListener('resize', handleResize);
//       }
//     };
//   }, [fileName]);

//   // ... (keep existing helper functions)
//     const processHierarchicalData = (rawData) => {
//     const nodes = [];
//     const links = [];
//     const nodeMap = new Map();
//     let nodeIndex = 0;

//     // Create root node
//     const rootNode = {
//       id: 'root',
//       label: 'Document Root',
//       level: 0,
//       type: 'ROOT',
//       color: '#FFFFFF',
//       size: 25,
//       children: [],
//       x: 0,  // Set initial position
//       y: 0,
//       z: 0,
//       index: nodeIndex++
//     };
//     nodes.push(rootNode);
//     nodeMap.set(rootNode.id, rootNode);

//     // First pass: Create all nodes and establish parent-child relationships
//     const levelMap = new Map(); // Track nodes by level
//     rawData.nodes.forEach(node => {
//       const level = getNodeLevel(node.type) || 1;
//       if (!levelMap.has(level)) {
//         levelMap.set(level, []);
//       }
//       levelMap.get(level).push(node.id);

//       const processedNode = {
//         ...node,
//         children: [],
//         level: level,
//         size: Math.max(15, (node.importance || 1) * 10),
//         color: getNodeColor(node.type),
//         index: nodeIndex++,
//         // Set initial positions in a circle based on level
//         x: Math.cos(nodeIndex * 0.1) * (level * 100),
//         y: Math.sin(nodeIndex * 0.1) * (level * 100),
//         z: level * 50
//       };
//       nodes.push(processedNode);
//       nodeMap.set(processedNode.id, processedNode);
//     });

//     // Second pass: Build hierarchy and connect to root
//     rawData.edges.forEach(edge => {
//       const source = nodeMap.get(edge.source);
//       const target = nodeMap.get(edge.target);
//       if (source && target) {
//         source.children.push(target);
//         target.parent = source.id; // Add parent reference
//         links.push({
//           source: source.id,
//           target: target.id,
//           value: edge.weight || 1
//         });
//       }
//     });

//     // Connect orphan nodes to root
//     nodes.forEach(node => {
//       if (node.id !== 'root' && !node.parent) {
//         rootNode.children.push(node);
//         node.parent = 'root';
//         links.push({
//           source: 'root',
//           target: node.id,
//           value: 1
//         });
//       }
//     });

//     // Calculate siblings count for each level
//     levelMap.forEach((nodeIds, level) => {
//       nodeIds.forEach((nodeId, index) => {
//         const node = nodeMap.get(nodeId);
//         if (node) {
//           node.siblings = nodeIds.length;
//           node.levelIndex = index;
//         }
//       });
//     });

//     return { nodes, links };
//   };

//   const getDescendants = (node, data) => {
//     const descendants = new Set();
//     const traverse = (n) => {
//       descendants.add(n.id);
//       (n.children || []).forEach(child => {
//         const childNode = data.nodes.find(node => node.id === child.id);
//         if (childNode) traverse(childNode);
//       });
//     };
//     traverse(node);
//     return descendants;
//   };

//   const updateVisibleNodes = (data, expanded) => {
//     try {
//       const visibleNodes = data.nodes.filter(node => {
//         // Always show root
//         if (node.id === 'root') return true;
//         // Show if node is expanded
//         if (expanded.has(node.id)) return true;
//         // Show if parent is expanded
//         if (node.parent && expanded.has(node.parent)) return true;
//         return false;
//       });

//       const visibleNodeIds = new Set(visibleNodes.map(node => node.id));
//       const visibleLinks = data.links.filter(link => 
//         visibleNodeIds.has(link.source) && visibleNodeIds.has(link.target)
//       );

//       if (graphRef.current) {
//         graphRef.current.graphData({
//           nodes: visibleNodes,
//           links: visibleLinks
//         });
//       }
//     } catch (error) {
//       console.error('Error updating visible nodes:', error);
//     }
//   };

//   // Updated color scheme for better visibility
//   const getNodeColor = (type) => {
//     switch (type) {
//       case 'MAIN_TOPIC':
//         return '#4CAF50'; // Bright green
//       case 'SUBTOPIC':
//         return '#2196F3'; // Bright blue
//       case 'CONCEPT':
//         return '#FFC107'; // Bright yellow
//       case 'ENTITY':
//         return '#E91E63'; // Bright pink
//       case 'METADATA':
//         return '#9C27B0'; // Bright purple
//       default:
//         return '#FF5722'; // Bright orange
//     }
//   };

//   const downloadCSV = () => { 
//     if (!graphRef.current) return;
    
//     const graphData = graphRef.current.graphData();
//     console.log(graphData);
//     const sanitizeText = (text) => {
//       if (!text) return 'null'; // Handle empty values
//       text = text.replace(/\u0000/g, ''); // Remove null characters
//       return `"'${text}"'`; // Force Excel to treat it as text
//   };
  
//   const nodesCSV = [
//       ['id', 'label', 'type', 'level', 'color'].join(','), // Header
//       ...graphData.nodes.map(node => [
//           `"${node.id}"`,
//           sanitizeText(node.label), // Apply sanitization
//           `"${node.type}"`,
//           node.level,
//           `"${node.color}"`
//       ].join(','))
//   ].join('\n');
  
//     // Create edges CSV
//     const edgesCSV = [
//         ['source', 'target', 'relationship', 'weight'].join(','), // Header
//         ...graphData.links.map(link => [
//             `"${link.source.id || link.source}"`,
//             `"${link.target.id || link.target}"`,
//             `"${link.label || ''}"`,
//             link.value || 1
//         ].join(','))
//     ].join('\n');
//     // Download function
//     const downloadFile = (content, fileName) => {
//         const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
//         const link = document.createElement('a');
//         const url = URL.createObjectURL(blob);
//         link.href = url;
//         link.download = fileName;
//         document.body.appendChild(link);
//         link.click();
//         document.body.removeChild(link);
//         URL.revokeObjectURL(url);
//     };

//     // Ensure fileName is provided
//     const baseFileName = fileName ? fileName.replace('.pdf', '') : 'graph_data';

//     // Download both files
//     downloadFile(nodesCSV, `${baseFileName}_nodes.csv`);
//     downloadFile(edgesCSV, `${baseFileName}_edges.csv`);
// };


//   // Add this helper function
//   const getNodeLevel = (type) => {
//     switch (type) {
//       case 'ROOT':
//         return 0;
//       case 'MAIN_TOPIC':
//         return 1;
//       case 'SUBTOPIC':
//         return 2;
//       case 'CONCEPT':
//         return 3;
//       case 'ENTITY':
//         return 4;
//       default:
//         return 2;
//     }
//   };
//   const closeGraph =()=>{navigate('/');}

//   return (
//     <div className="knowledge-graph-container">
//       <div className="graph-header">
//         <h3>Knowledge Graph: {fileName}</h3>
//         <div className="search-filter-container">
//           <div className="search-bar">
//             <input
//               type="text"
//               placeholder="Search nodes..."
//               value={searchTerm}
//               onChange={handleSearchChange}
//               className="search-input"
//             />
//           </div>
//           <div className="filter-options">
//             {nodeTypes.map(type => (
//               <label key={type.value} className="filter-checkbox">
//                 <input
//                   type="checkbox"
//                   checked={selectedNodeTypes.includes(type.value)}
//                   onChange={() => handleNodeTypeChange(type.value)}
//                 />
//                 {type.label}
//               </label>
//             ))}
//           </div>
//         </div>
//         <div className="graph-controls">
//           <button 
//             onClick={downloadCSV}
//             className="download-btn"
//             disabled={loading || error}
//           >
//             Download CSV
//           </button>
//           <button onClick={closeGraph}>Close</button>
//         </div>
//       </div>
//       <div className="graph-content">
//         {loading && <div className="graph-loading">Loading graph...</div>}
//         {error && <div className="graph-error">Error: {error}</div>}
//         <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
//       </div>
//     </div>
//   );
// }

// export default KnowledgeGraph;
// import React, { useEffect, useRef, useState } from 'react';
// import ForceGraph3D from '3d-force-graph';
// import * as THREE from 'three';
// import { 
//   forceSimulation,
//   forceManyBody,
//   forceLink,
//   forceCenter,
//   forceY,
//   forceCollide
// } from 'd3-force-3d';
// import '../styles/KnowledgeGraph.css';

// // Custom SpriteText class
// class SpriteText extends THREE.Sprite {
//   constructor(text, textHeight = 8, color = '#ffffff', backgroundColor = 'rgba(0,0,0,0.8)') {
//     super(new THREE.SpriteMaterial({ map: new THREE.Texture() }));
//     this.text = text;
//     this.textHeight = textHeight;
//     this.color = color;
//     this.backgroundColor = backgroundColor;
//     this.padding = 2;
//     this.updateTexture();
//   }

//   updateTexture() {
//     const canvas = document.createElement('canvas');
//     const ctx = canvas.getContext('2d');
//     ctx.font = `${this.textHeight}px Arial`;
    

//     const textWidth = ctx.measureText(this.text).width;
//     canvas.width = textWidth + (this.padding * 2);
//     canvas.height = this.textHeight + (this.padding * 2);
    
//     // Draw background
//     ctx.fillStyle = this.backgroundColor;
//     ctx.fillRect(0, 0, canvas.width, canvas.height);
    
//     // Draw text
//     ctx.font = `${this.textHeight}px Arial`;
//     ctx.fillStyle = this.color;
//     ctx.textBaseline = 'middle';
//     ctx.textAlign = 'center';
//     ctx.fillText(this.text, canvas.width / 2, canvas.height / 2);
    

//     const texture = new THREE.Texture(canvas);
//     texture.needsUpdate = true;
//     this.material.map = texture;
//     this.scale.set(canvas.width / canvas.height * this.textHeight, this.textHeight, 1);
//   }
// }

// function KnowledgeGraph({ fileName, onClose }) {
//   const containerRef = useRef();
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);
//   const graphRef = useRef(null);
//   const [zoomLevel, setZoomLevel] = useState(1);
//   const [expandedNodes, setExpandedNodes] = useState(new Set(['root']));

//   useEffect(() => {
//     let handleResize;

//     const initGraph = async () => {
//       if (!fileName || !containerRef.current) return;

//       try {
//         setLoading(true);
//         const response = await fetch(`http://localhost:5000/api/knowledge-graph?file_name=${fileName}`);
//         if (!response.ok) throw new Error('Failed to fetch knowledge graph data');
        

//         const data = await response.json();
//         const graphData = data.graph;

//         // Process data into hierarchical structure
//         const processedData = processHierarchicalData(graphData);

//         // Initialize force graph with radial layout
//         graphRef.current = ForceGraph3D()(containerRef.current)
//           .graphData(processedData)
//           .backgroundColor('#000000')
//           .nodeLabel('label')
//           .nodeColor(node => node.color)
//           .nodeRelSize(8)
//           .nodeVal(node => node.size)
//           .width(window.innerWidth)
//           .height(window.innerHeight - 60)
//           .linkWidth(2)
//           .linkOpacity(0.5)
//           .nodeThreeObject(node => {
//             const group = new THREE.Group();
            
//             // Scale node size based on level and importance
//             const baseRadius = Math.max(5, 20 - node.level * 3);
//             const radius = baseRadius * (node.importance || 1);
            
//             // Create node sphere
//             const geometry = new THREE.SphereGeometry(radius);
//             const material = new THREE.MeshPhongMaterial({
//               color: node.color,
//               transparent: true,
//               opacity: 0.8,
//               shininess: 100,
//               emissive: node.color,
//               emissiveIntensity: 0.2
//             });
//             const sphere = new THREE.Mesh(geometry, material);
//             group.add(sphere);

//             // Add label if node is expanded or important
//             if (expandedNodes.has(node.id) || node.level <= 1) {
//               const sprite = new SpriteText(
//                 node.label,
//                 radius * 2,
//                 '#ffffff',
//                 'rgba(0,0,0,0.8)'
//               );
//               sprite.position.y = radius + 2;
//               sprite.fontWeight = 'bold';
//               group.add(sprite);
//             }

//             return group;
//           })
//           .onNodeClick(node => {
//             // Toggle node expansion
//             const newExpanded = new Set(expandedNodes);
//             if (newExpanded.has(node.id)) {
//               // Collapse node and its children
//               const toRemove = getDescendants(node, processedData);
//               toRemove.forEach(id => newExpanded.delete(id));
//             } else {
//               // Expand node
//               newExpanded.add(node.id);
//             }
//             setExpandedNodes(newExpanded);
            
//             // Update visible nodes and links
//             updateVisibleNodes(processedData, newExpanded);
            
//             // Center view on clicked node
//             graphRef.current.cameraPosition(
//               { x: node.x * 1.4, y: node.y * 1.4, z: node.z * 1.4 },
//               node,
//               2000
//             );
//           });

//         // Configure forces for radial layout
//         const radialForce = alpha => {
//           processedData.nodes.forEach(node => {
//             if (!node.x && !node.y) return; // Skip nodes without position

//             // Calculate ideal radial position based on level
//             const angle = (node.index || 0) * (2 * Math.PI / (node.siblings || 1));
//             const radius = node.level * 200; // Increase radius for each level
            
//             const targetX = Math.cos(angle) * radius;
//             const targetY = Math.sin(angle) * radius;
//             const targetZ = node.level * 50; // Slight vertical separation

//             // Apply force towards ideal position
//             node.vx += (targetX - node.x) * alpha * 0.1;
//             node.vy += (targetY - node.y) * alpha * 0.1;
//             node.vz += (targetZ - node.z) * alpha * 0.1;
//           });
//         };

//         graphRef.current
//           .d3Force('link', forceLink()
//             .id(d => d.id)
//             .distance(link => 100 + link.source.level * 50))
//           .d3Force('charge', forceManyBody().strength(-1000))
//           .d3Force('center', forceCenter())
//           .d3Force('radial', radialForce)
//           .d3Force('collision', forceCollide(node => 30 + node.level * 10));

//         // Add lights
//         const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
//         graphRef.current.scene().add(ambientLight);

//         const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
//         directionalLight.position.set(200, 200, 200);
//         graphRef.current.scene().add(directionalLight);

//         setLoading(false);
//       } catch (err) {
//         console.error('Error initializing graph:', err);
//         setError(err.message);
//         setLoading(false);
//       }
//     };

//     initGraph();

//     return () => {
//       if (graphRef.current) {
//         graphRef.current._destructor();
//       }
//       if (handleResize) {
//         window.removeEventListener('resize', handleResize);
//       }
//     };
//   }, [fileName, expandedNodes]);

//   // Helper functions
//   const processHierarchicalData = (rawData) => {
//     const nodes = [];
//     const links = [];
//     const nodeMap = new Map();
//     let nodeIndex = 0;

//     // Create root node
//     const rootNode = {
//       id: 'root',
//       label: 'Document Root',
//       level: 0,
//       type: 'ROOT',
//       color: '#FFFFFF',
//       size: 25,
//       children: [],
//       x: 0,  // Set initial position
//       y: 0,
//       z: 0,
//       index: nodeIndex++
//     };
//     nodes.push(rootNode);
//     nodeMap.set(rootNode.id, rootNode);

//     // First pass: Create all nodes and establish parent-child relationships
//     const levelMap = new Map(); // Track nodes by level
//     rawData.nodes.forEach(node => {
//       const level = getNodeLevel(node.type) || 1;
//       if (!levelMap.has(level)) {
//         levelMap.set(level, []);
//       }
//       levelMap.get(level).push(node.id);

//       const processedNode = {
//         ...node,
//         children: [],
//         level: level,
//         size: Math.max(15, (node.importance || 1) * 10),
//         color: getNodeColor(node.type),
//         index: nodeIndex++,
//         // Set initial positions in a circle based on level
//         x: Math.cos(nodeIndex * 0.1) * (level * 100),
//         y: Math.sin(nodeIndex * 0.1) * (level * 100),
//         z: level * 50
//       };
//       nodes.push(processedNode);
//       nodeMap.set(processedNode.id, processedNode);
//     });

//     // Second pass: Build hierarchy and connect to root
//     rawData.edges.forEach(edge => {
//       const source = nodeMap.get(edge.source);
//       const target = nodeMap.get(edge.target);
//       if (source && target) {
//         source.children.push(target);
//         target.parent = source.id; // Add parent reference
//         links.push({
//           source: source.id,
//           target: target.id,
//           value: edge.weight || 1
//         });
//       }
//     });

//     // Connect orphan nodes to root
//     nodes.forEach(node => {
//       if (node.id !== 'root' && !node.parent) {
//         rootNode.children.push(node);
//         node.parent = 'root';
//         links.push({
//           source: 'root',
//           target: node.id,
//           value: 1
//         });
//       }
//     });

//     // Calculate siblings count for each level
//     levelMap.forEach((nodeIds, level) => {
//       nodeIds.forEach((nodeId, index) => {
//         const node = nodeMap.get(nodeId);
//         if (node) {
//           node.siblings = nodeIds.length;
//           node.levelIndex = index;
//         }
//       });
//     });

//     return { nodes, links };
//   };

//   const getDescendants = (node, data) => {
//     const descendants = new Set();
//     const traverse = (n) => {
//       descendants.add(n.id);
//       (n.children || []).forEach(child => {
//         const childNode = data.nodes.find(node => node.id === child.id);
//         if (childNode) traverse(childNode);
//       });
//     };
//     traverse(node);
//     return descendants;
//   };

//   const updateVisibleNodes = (data, expanded) => {
//     try {
//       const visibleNodes = data.nodes.filter(node => {
//         // Always show root
//         if (node.id === 'root') return true;
//         // Show if node is expanded
//         if (expanded.has(node.id)) return true;
//         // Show if parent is expanded
//         if (node.parent && expanded.has(node.parent)) return true;
//         return false;
//       });

//       const visibleNodeIds = new Set(visibleNodes.map(node => node.id));
//       const visibleLinks = data.links.filter(link => 
//         visibleNodeIds.has(link.source) && visibleNodeIds.has(link.target)
//       );

//       if (graphRef.current) {
//         graphRef.current.graphData({
//           nodes: visibleNodes,
//           links: visibleLinks
//         });
//       }
//     } catch (error) {
//       console.error('Error updating visible nodes:', error);
//     }
//   };

//   // Updated color scheme for better visibility
//   const getNodeColor = (type) => {
//     switch (type) {
//       case 'MAIN_TOPIC':
//         return '#4CAF50'; // Bright green
//       case 'SUBTOPIC':
//         return '#2196F3'; // Bright blue
//       case 'CONCEPT':
//         return '#FFC107'; // Bright yellow
//       case 'ENTITY':
//         return '#E91E63'; // Bright pink
//       case 'METADATA':
//         return '#9C27B0'; // Bright purple
//       default:
//         return '#FF5722'; // Bright orange
//     }
//   };

//   const downloadCSV = () => { 
//     if (!graphRef.current) return;
    
//     const graphData = graphRef.current.graphData();
//     console.log(graphData);
//     const sanitizeText = (text) => {
//       if (!text) return 'null'; // Handle empty values
//       text = text.replace(/\u0000/g, ''); // Remove null characters
//       return `"'${text}"'`; // Force Excel to treat it as text
//   };
  
//   const nodesCSV = [
//       ['id', 'label', 'type', 'level', 'color'].join(','), // Header
//       ...graphData.nodes.map(node => [
//           `"${node.id}"`,
//           sanitizeText(node.label), // Apply sanitization
//           `"${node.type}"`,
//           node.level,
//           `"${node.color}"`
//       ].join(','))
//   ].join('\n');
  
//     // Create edges CSV
//     const edgesCSV = [
//         ['source', 'target', 'relationship', 'weight'].join(','), // Header
//         ...graphData.links.map(link => [
//             `"${link.source.id || link.source}"`,
//             `"${link.target.id || link.target}"`,
//             `"${link.label || ''}"`,
//             link.value || 1
//         ].join(','))
//     ].join('\n');
//     // Download function
//     const downloadFile = (content, fileName) => {
//         const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
//         const link = document.createElement('a');
//         const url = URL.createObjectURL(blob);
//         link.href = url;
//         link.download = fileName;
//         document.body.appendChild(link);
//         link.click();
//         document.body.removeChild(link);
//         URL.revokeObjectURL(url);
//     };

//     // Ensure fileName is provided
//     const baseFileName = fileName ? fileName.replace('.pdf', '') : 'graph_data';

//     // Download both files
//     downloadFile(nodesCSV, `${baseFileName}_nodes.csv`);
//     downloadFile(edgesCSV, `${baseFileName}_edges.csv`);
// };


//   // Add this helper function
//   const getNodeLevel = (type) => {
//     switch (type) {
//       case 'ROOT':
//         return 0;
//       case 'MAIN_TOPIC':
//         return 1;
//       case 'SUBTOPIC':
//         return 2;
//       case 'CONCEPT':
//         return 3;
//       case 'ENTITY':
//         return 4;
//       default:
//         return 2;
//     }
//   };

//   return (
//     <div className="knowledge-graph-container">
//       <div className="graph-header">
//         <h3>Knowledge Graph: {fileName}</h3>
//         <div className="graph-controls">
//           <button 
//             onClick={downloadCSV}
//             className="download-btn"
//             disabled={loading || error}
//           >
//             Download CSV
//           </button>
//           <button onClick={onClose}>Close</button>
//         </div>
//       </div>
//       <div className="graph-content">
//         {loading && <div className="graph-loading">Loading graph...</div>}
//         {error && <div className="graph-error">Error: {error}</div>}
//         <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
//       </div>
//     </div>
//   );
// }

// export default KnowledgeGraph;
