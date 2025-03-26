// src/services/visualizationService.js

// Function to fetch and process embeddings from the backend
export const fetchAndProcessEmbeddings = async (fileName) => {
  try {
    const response = await fetch(`http://localhost:5000/api/embeddings?file_name=${fileName}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.embeddings;
  } catch (error) {
    console.error('Error fetching embeddings:', error);
    return [];
  }
};

// Function to reduce high-dimensional embeddings to 3D for visualization
// This is a simplified implementation of t-SNE or UMAP
export const reduceToThreeDimensions = (embeddings) => {
  // For demo purposes, we'll create a simple projection
  // In a real implementation, you would use a proper dimensionality reduction algorithm
  return embeddings.map(embedding => {
    // Create a 3D projection from the embedding vector
    // This is just a placeholder - real implementation would use t-SNE or UMAP
    const sumGroups = [];
    const groupSize = Math.ceil(embedding.length / 3);
    
    for (let i = 0; i < 3; i++) {
      const start = i * groupSize;
      const end = Math.min(start + groupSize, embedding.length);
      const sum = embedding.slice(start, end).reduce((acc, val) => acc + val, 0);
      sumGroups.push(sum);
    }
    
    // Normalize to a reasonable range
    const max = Math.max(...sumGroups);
    const min = Math.min(...sumGroups);
    const range = max - min || 1;
    
    return sumGroups.map(val => (val - min) / range * 10 - 5);
  });
};

// Function to cluster the 3D embeddings
export const clusterEmbeddings = (reducedData) => {
  // Simple clustering based on quadrants in 3D space
  // In a real implementation, you would use k-means or another clustering algorithm
  return reducedData.map(point => {
    const x = point[0] > 0 ? 1 : 0;
    const y = point[1] > 0 ? 1 : 0;
    const z = point[2] > 0 ? 1 : 0;
    
    // This gives 8 possible clusters (2^3)
    return x * 4 + y * 2 + z;
  });
};
// import {UMAP} from 'umap-js';

// export async function fetchAndProcessEmbeddings(fileName) {
//   try {
//     const response = await fetch(`http://localhost:5000/api/embeddings?file_name=${fileName}`);
//     const data = await response.json();
//     console.log('Fetched embeddings:', data.embeddings.length);
//     return data.embeddings;
//   } catch (error) {
//     console.error('Error fetching embeddings:', error);
//     return null;
//   }
// }

// export function normalizeEmbeddings(embeddings) {
//   if (!embeddings || embeddings.length === 0) return [];
  
//   console.log('Normalizing embeddings:', embeddings.length);
  
//   const maxValues = embeddings[0].map(() => -Infinity);
//   const minValues = embeddings[0].map(() => Infinity);

//   embeddings.forEach(embedding => {
//     embedding.forEach((value, i) => {
//       maxValues[i] = Math.max(maxValues[i], value);
//       minValues[i] = Math.min(minValues[i], value);
//     });
//   });

//   const normalized = embeddings.map(embedding =>
//     embedding.map((value, i) => {
//       const range = maxValues[i] - minValues[i];
//       return range === 0 ? 0 : (2 * (value - minValues[i]) / range) - 1;
//     })
//   );

//   console.log('Normalized embeddings sample:', normalized[0]);
//   return normalized;
// }

// export function reduceToThreeDimensions(embeddings) {
//   try {
//     const normalizedEmbeddings = normalizeEmbeddings(embeddings);
//     if (normalizedEmbeddings.length === 0) return [];

//     console.log('Reducing dimensions for', normalizedEmbeddings.length, 'points');

//     const umap = new UMAP({
//       nComponents: 3,
//       nNeighbors: 2,
//       minDist: 0.1,
//       spread: 1.0,
//       random: Math.random,
//     });
    
//     const data = normalizedEmbeddings.map(embedding => 
//       Float64Array.from(embedding)
//     );
    
//     const reduced = umap.fit(data);
//     console.log('Reduced dimensions sample:', reduced[0]);
//     return reduced;
//   } catch (error) {
//     console.error('Error in dimension reduction:', error);
//     return [];
//   }
// }

// export function clusterEmbeddings(reducedData, numClusters = 5) {
//   if (!reducedData || reducedData.length === 0) return [];
  
//   console.log('Clustering', reducedData.length, 'points');
//   const clusters = new Array(reducedData.length).fill(0);
//   reducedData.forEach((_, i) => {
//     clusters[i] = Math.floor(Math.random() * numClusters);
//   });
//   return clusters;
// } 