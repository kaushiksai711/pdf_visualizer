import React, { useEffect, useRef, useState } from 'react';
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

// // Custom SpriteText class
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

function KnowledgeGraph({ fileName, onClose }) {
  const containerRef = useRef();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const graphRef = useRef(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [expandedNodes, setExpandedNodes] = useState(new Set(['root']));
  
  // New states for search and filtering
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedNodeTypes, setSelectedNodeTypes] = useState(['MAIN_TOPIC', 'SUBTOPIC', 'CONCEPT', 'ENTITY', 'METADATA']);
  const [originalGraphData, setOriginalGraphData] = useState(null);
  const [filteredGraphData, setFilteredGraphData] = useState(null);

  // Node type options for the filter
  const nodeTypes = [
    { value: 'MAIN_TOPIC', label: 'Main Topics' },
    { value: 'SUBTOPIC', label: 'Subtopics' },
    { value: 'CONCEPT', label: 'Concepts' },
    { value: 'ENTITY', label: 'Entities' },
    { value: 'METADATA', label: 'Metadata' }
  ];

  // Search and filter function
  const filterGraph = () => {
    if (!originalGraphData) return;

    const searchLower = searchTerm.toLowerCase();
    
    // Filter nodes based on search term and selected types
    const filteredNodes = originalGraphData.nodes.filter(node => {
      const matchesSearch = node.label?.toLowerCase().includes(searchLower) || searchTerm === '';
      const matchesType = selectedNodeTypes.includes(node.type) || node.type === 'ROOT';
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
      graphRef.current.graphData({ nodes: filteredNodes, links: filteredLinks });
    }
  };

  // Handle search input change
  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
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

  // Effect to apply filters when search term or node types change
  useEffect(() => {
    filterGraph();
  }, [searchTerm, selectedNodeTypes]);

  useEffect(() => {
    let handleResize;

    const initGraph = async () => {
      if (!fileName || !containerRef.current) return;

      try {
        setLoading(true);
        const response = await fetch(`http://localhost:5000/api/knowledge-graph?file_name=${fileName}`);
        if (!response.ok) throw new Error('Failed to fetch knowledge graph data');

        const data = await response.json();
        const graphData = data.graph;

        // Process data into hierarchical structure
        const processedData = processHierarchicalData(graphData);
        setOriginalGraphData(processedData);
        setFilteredGraphData(processedData);

        // Initialize force graph with radial layout
        
//         // Initialize force graph with radial layout
        graphRef.current = ForceGraph3D()(containerRef.current)
          .graphData(processedData)
          .backgroundColor('#000000')
          .nodeLabel('label')
          .nodeColor(node => node.color)
          .nodeRelSize(8)
          .nodeVal(node => node.size)
          .width(window.innerWidth)
          .height(window.innerHeight - 60)
          .linkWidth(2)
          .linkOpacity(0.5)
          .nodeThreeObject(node => {
            const group = new THREE.Group();
            
            // Scale node size based on level and importance
            const baseRadius = Math.max(5, 20 - node.level * 3);
            const radius = baseRadius * (node.importance || 1);
            
            // Create node sphere
            const geometry = new THREE.SphereGeometry(radius);
            const material = new THREE.MeshPhongMaterial({
              color: node.color,
              transparent: true,
              opacity: 0.8,
              shininess: 100,
              emissive: node.color,
              emissiveIntensity: 0.2
            });
            const sphere = new THREE.Mesh(geometry, material);
            group.add(sphere);

            // Add label if node is expanded or important
            if (expandedNodes.has(node.id) || node.level <= 1) {
              const sprite = new SpriteText(
                node.label,
                radius * 2,
                '#ffffff',
                'rgba(0,0,0,0.8)'
              );
              sprite.position.y = radius + 2;
              sprite.fontWeight = 'bold';
              group.add(sprite);
            }

            return group;
          })
          .onNodeClick(node => {
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
            
            // Center view on clicked node
            graphRef.current.cameraPosition(
              { x: node.x * 1.4, y: node.y * 1.4, z: node.z * 1.4 },
              node,
              2000
            );
          });

        // Configure forces for radial layout
        const radialForce = alpha => {
          processedData.nodes.forEach(node => {
            if (!node.x && !node.y) return; // Skip nodes without position

            // Calculate ideal radial position based on level
            const angle = (node.index || 0) * (2 * Math.PI / (node.siblings || 1));
            const radius = node.level * 200; // Increase radius for each level
            
            const targetX = Math.cos(angle) * radius;
            const targetY = Math.sin(angle) * radius;
            const targetZ = node.level * 50; // Slight vertical separation

            // Apply force towards ideal position
            node.vx += (targetX - node.x) * alpha * 0.1;
            node.vy += (targetY - node.y) * alpha * 0.1;
            node.vz += (targetZ - node.z) * alpha * 0.1;
          });
        };

        graphRef.current
          .d3Force('link', forceLink()
            .id(d => d.id)
            .distance(link => 100 + link.source.level * 50))
          .d3Force('charge', forceManyBody().strength(-1000))
          .d3Force('center', forceCenter())
          .d3Force('radial', radialForce)
          .d3Force('collision', forceCollide(node => 30 + node.level * 10));

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        graphRef.current.scene().add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(200, 200, 200);
        graphRef.current.scene().add(directionalLight);

        setLoading(false);

      } catch (err) {
        console.error('Error initializing graph:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    initGraph();

    return () => {
      if (graphRef.current) {
        graphRef.current._destructor();
      }
      if (handleResize) {
        window.removeEventListener('resize', handleResize);
      }
    };
  }, [fileName]);

  // ... (keep existing helper functions)
    const processHierarchicalData = (rawData) => {
    const nodes = [];
    const links = [];
    const nodeMap = new Map();
    let nodeIndex = 0;

    // Create root node
    const rootNode = {
      id: 'root',
      label: 'Document Root',
      level: 0,
      type: 'ROOT',
      color: '#FFFFFF',
      size: 25,
      children: [],
      x: 0,  // Set initial position
      y: 0,
      z: 0,
      index: nodeIndex++
    };
    nodes.push(rootNode);
    nodeMap.set(rootNode.id, rootNode);

    // First pass: Create all nodes and establish parent-child relationships
    const levelMap = new Map(); // Track nodes by level
    rawData.nodes.forEach(node => {
      const level = getNodeLevel(node.type) || 1;
      if (!levelMap.has(level)) {
        levelMap.set(level, []);
      }
      levelMap.get(level).push(node.id);

      const processedNode = {
        ...node,
        children: [],
        level: level,
        size: Math.max(15, (node.importance || 1) * 10),
        color: getNodeColor(node.type),
        index: nodeIndex++,
        // Set initial positions in a circle based on level
        x: Math.cos(nodeIndex * 0.1) * (level * 100),
        y: Math.sin(nodeIndex * 0.1) * (level * 100),
        z: level * 50
      };
      nodes.push(processedNode);
      nodeMap.set(processedNode.id, processedNode);
    });

    // Second pass: Build hierarchy and connect to root
    rawData.edges.forEach(edge => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (source && target) {
        source.children.push(target);
        target.parent = source.id; // Add parent reference
        links.push({
          source: source.id,
          target: target.id,
          value: edge.weight || 1
        });
      }
    });

    // Connect orphan nodes to root
    nodes.forEach(node => {
      if (node.id !== 'root' && !node.parent) {
        rootNode.children.push(node);
        node.parent = 'root';
        links.push({
          source: 'root',
          target: node.id,
          value: 1
        });
      }
    });

    // Calculate siblings count for each level
    levelMap.forEach((nodeIds, level) => {
      nodeIds.forEach((nodeId, index) => {
        const node = nodeMap.get(nodeId);
        if (node) {
          node.siblings = nodeIds.length;
          node.levelIndex = index;
        }
      });
    });

    return { nodes, links };
  };

  const getDescendants = (node, data) => {
    const descendants = new Set();
    const traverse = (n) => {
      descendants.add(n.id);
      (n.children || []).forEach(child => {
        const childNode = data.nodes.find(node => node.id === child.id);
        if (childNode) traverse(childNode);
      });
    };
    traverse(node);
    return descendants;
  };

  const updateVisibleNodes = (data, expanded) => {
    try {
      const visibleNodes = data.nodes.filter(node => {
        // Always show root
        if (node.id === 'root') return true;
        // Show if node is expanded
        if (expanded.has(node.id)) return true;
        // Show if parent is expanded
        if (node.parent && expanded.has(node.parent)) return true;
        return false;
      });

      const visibleNodeIds = new Set(visibleNodes.map(node => node.id));
      const visibleLinks = data.links.filter(link => 
        visibleNodeIds.has(link.source) && visibleNodeIds.has(link.target)
      );

      if (graphRef.current) {
        graphRef.current.graphData({
          nodes: visibleNodes,
          links: visibleLinks
        });
      }
    } catch (error) {
      console.error('Error updating visible nodes:', error);
    }
  };

  // Updated color scheme for better visibility
  const getNodeColor = (type) => {
    switch (type) {
      case 'MAIN_TOPIC':
        return '#4CAF50'; // Bright green
      case 'SUBTOPIC':
        return '#2196F3'; // Bright blue
      case 'CONCEPT':
        return '#FFC107'; // Bright yellow
      case 'ENTITY':
        return '#E91E63'; // Bright pink
      case 'METADATA':
        return '#9C27B0'; // Bright purple
      default:
        return '#FF5722'; // Bright orange
    }
  };

  const downloadCSV = () => { 
    if (!graphRef.current) return;
    
    const graphData = graphRef.current.graphData();
    console.log(graphData);
    const sanitizeText = (text) => {
      if (!text) return 'null'; // Handle empty values
      text = text.replace(/\u0000/g, ''); // Remove null characters
      return `"'${text}"'`; // Force Excel to treat it as text
  };
  
  const nodesCSV = [
      ['id', 'label', 'type', 'level', 'color'].join(','), // Header
      ...graphData.nodes.map(node => [
          `"${node.id}"`,
          sanitizeText(node.label), // Apply sanitization
          `"${node.type}"`,
          node.level,
          `"${node.color}"`
      ].join(','))
  ].join('\n');
  
    // Create edges CSV
    const edgesCSV = [
        ['source', 'target', 'relationship', 'weight'].join(','), // Header
        ...graphData.links.map(link => [
            `"${link.source.id || link.source}"`,
            `"${link.target.id || link.target}"`,
            `"${link.label || ''}"`,
            link.value || 1
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


  // Add this helper function
  const getNodeLevel = (type) => {
    switch (type) {
      case 'ROOT':
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
        return 2;
    }
  };

  return (
    <div className="knowledge-graph-container">
      <div className="graph-header">
        <h3>Knowledge Graph: {fileName}</h3>
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
          <button 
            onClick={downloadCSV}
            className="download-btn"
            disabled={loading || error}
          >
            Download CSV
          </button>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
      <div className="graph-content">
        {loading && <div className="graph-loading">Loading graph...</div>}
        {error && <div className="graph-error">Error: {error}</div>}
        <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
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