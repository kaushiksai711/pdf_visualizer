export async function fetchKnowledgeGraph(fileName) {
  try {
    const response = await fetch(`http://localhost:5000/api/knowledge-graph?file_name=${fileName}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    console.log('Raw graph data:', data); // Debug log
    return data.graph;
  } catch (error) {
    console.error('Error fetching knowledge graph:', error);
    return null;
  }
}

export function processGraphData(graphData) {
  if (!graphData || !graphData.nodes || !graphData.edges) {
    console.error('Invalid graph data:', graphData);
    return { nodes: [], edges: [] };
  }

  // Create a map of node IDs to array indices
  const nodeIdMap = new Map();
  const nodes = graphData.nodes.map((node, index) => {
    nodeIdMap.set(node.id, index);
    return {
      id: index,
      label: node.label || '',
      type: node.type || 'DEFAULT',
      size: Math.max((node.importance || 0.5) * 5, 2), // Ensure minimum size
      color: (() => {
        switch(node.type) {
          case 'PERSON': return '#ff6b6b';
          case 'ORG': return '#4ecdc4';
          case 'GPE': return '#45b7d1';
          default: return '#96a5d1';
        }
      })()
    };
  });

  // Map edge references to node indices
  const edges = graphData.edges
    .filter(edge => 
      nodeIdMap.has(edge.source) && 
      nodeIdMap.has(edge.target)
    )
    .map(edge => ({
      source: nodeIdMap.get(edge.source),
      target: nodeIdMap.get(edge.target),
      label: edge.relationship || '',
      value: 1 // Default strength
    }));

  console.log('Processed data:', { nodes, edges });
  return { nodes, edges };
} 