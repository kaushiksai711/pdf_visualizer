import {UMAP} from 'umap-js';

export async function fetchAndProcessEmbeddings(fileName) {
  try {
    const response = await fetch(`http://localhost:5000/api/embeddings?file_name=${fileName}`);
    const data = await response.json();
    console.log('Fetched embeddings:', data.embeddings.length);
    return data.embeddings;
  } catch (error) {
    console.error('Error fetching embeddings:', error);
    return null;
  }
}

export function normalizeEmbeddings(embeddings) {
  if (!embeddings || embeddings.length === 0) return [];
  
  console.log('Normalizing embeddings:', embeddings.length);
  
  const maxValues = embeddings[0].map(() => -Infinity);
  const minValues = embeddings[0].map(() => Infinity);

  embeddings.forEach(embedding => {
    embedding.forEach((value, i) => {
      maxValues[i] = Math.max(maxValues[i], value);
      minValues[i] = Math.min(minValues[i], value);
    });
  });

  const normalized = embeddings.map(embedding =>
    embedding.map((value, i) => {
      const range = maxValues[i] - minValues[i];
      return range === 0 ? 0 : (2 * (value - minValues[i]) / range) - 1;
    })
  );

  console.log('Normalized embeddings sample:', normalized[0]);
  return normalized;
}

export function reduceToThreeDimensions(embeddings) {
  try {
    const normalizedEmbeddings = normalizeEmbeddings(embeddings);
    if (normalizedEmbeddings.length === 0) return [];

    console.log('Reducing dimensions for', normalizedEmbeddings.length, 'points');

    const umap = new UMAP({
      nComponents: 3,
      nNeighbors: 2,
      minDist: 0.1,
      spread: 1.0,
      random: Math.random,
    });
    
    const data = normalizedEmbeddings.map(embedding => 
      Float64Array.from(embedding)
    );
    
    const reduced = umap.fit(data);
    console.log('Reduced dimensions sample:', reduced[0]);
    return reduced;
  } catch (error) {
    console.error('Error in dimension reduction:', error);
    return [];
  }
}

export function clusterEmbeddings(reducedData, numClusters = 5) {
  if (!reducedData || reducedData.length === 0) return [];
  
  console.log('Clustering', reducedData.length, 'points');
  const clusters = new Array(reducedData.length).fill(0);
  reducedData.forEach((_, i) => {
    clusters[i] = Math.floor(Math.random() * numClusters);
  });
  return clusters;
} 