import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { 
  fetchAndProcessEmbeddings, 
  reduceToThreeDimensions,
  clusterEmbeddings 
} from '../services/visualizationService';
import '../styles/RagVisualizer.css';
import { Raycaster, SphereGeometry, MeshPhongMaterial, Mesh, PointLight } from 'three';

function RagVisualizer({ isExpanded, onToggleExpand, selectedDocs }) {
  const containerRef = useRef(null);
  const [visualizationData, setVisualizationData] = useState(null);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [level, setLevel] = useState(1);
  const [hoveredChunk, setHoveredChunk] = useState(null);
  const [tooltip, setTooltip] = useState({ show: false, text: '', x: 0, y: 0 });
  const [selectedChunkDetails, setSelectedChunkDetails] = useState(null);

  useEffect(() => {
    let scene, camera, renderer, controls, raycaster, mouse;
    let spheres = [];

    const init = async () => {
      if (!containerRef.current || !selectedDocs.length) return;

      // Setup Three.js scene
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf0f0f0);
      
      camera = new THREE.PerspectiveCamera(
        75,
        containerRef.current.clientWidth / containerRef.current.clientHeight,
        0.1,
        1000
      );
      
      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
      containerRef.current.appendChild(renderer.domElement);

      // Improved lighting setup
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(10, 10, 10);
      scene.add(directionalLight);

      // Add helper axes for debugging
      const axesHelper = new THREE.AxesHelper(5);
      scene.add(axesHelper);

      // Initialize raycaster and mouse
      raycaster = new Raycaster();
      mouse = new THREE.Vector2();

      // Add controls with better initial position
      controls = new OrbitControls(camera, renderer.domElement);
      camera.position.set(5, 5, 5);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;

      try {
        const embeddings = await fetchAndProcessEmbeddings(selectedDocs[0].name);
        if (!embeddings || embeddings.length === 0) return;

        const reducedData = reduceToThreeDimensions(embeddings);
        const clusters = clusterEmbeddings(reducedData);
        setVisualizationData({ reducedData, clusters });

        // Create larger, more visible spheres
        const sphereGeometry = new SphereGeometry(0.3, 32, 32);
        const clusterColors = [
          0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff
        ];

        // Calculate bounds for scaling
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;

        reducedData.forEach(point => {
          minX = Math.min(minX, point[0]);
          maxX = Math.max(maxX, point[0]);
          minY = Math.min(minY, point[1]);
          maxY = Math.max(maxY, point[1]);
          minZ = Math.min(minZ, point[2]);
          maxZ = Math.max(maxZ, point[2]);
        });

        // Scale factor to normalize positions
        const scale = 5 / Math.max(maxX - minX, maxY - minY, maxZ - minZ);

        reducedData.forEach((point, i) => {
          const material = new MeshPhongMaterial({
            color: clusterColors[clusters[i] % clusterColors.length],
            shininess: 30,
            specular: 0x444444
          });

          const sphere = new Mesh(sphereGeometry, material);
          
          // Scale and center the positions
          sphere.position.set(
            (point[0] - (maxX + minX)/2) * scale,
            (point[1] - (maxY + minY)/2) * scale,
            (point[2] - (maxZ + minZ)/2) * scale
          );

          sphere.userData = {
            chunkId: `Chunk ${i + 1}`,
            chunkIndex: i
          };
          spheres.push(sphere);
          scene.add(sphere);
        });

        // Center camera on the spheres
        const center = new THREE.Vector3();
        spheres.forEach(sphere => {
          center.add(sphere.position);
        });
        center.divideScalar(spheres.length);
        
        controls.target.copy(center);
        camera.position.set(
          center.x + 10,
          center.y + 10,
          center.z + 10
        );
        controls.update();

        // Event listeners for interaction
        const onMouseMove = (event) => {
          const rect = renderer.domElement.getBoundingClientRect();
          mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

          raycaster.setFromCamera(mouse, camera);
          const intersects = raycaster.intersectObjects(spheres);

          if (intersects.length > 0) {
            const intersected = intersects[0].object;
            setHoveredChunk(intersected);
            setTooltip({
              show: true,
              text: intersected.userData.chunkId,
              x: event.clientX,
              y: event.clientY
            });
            document.body.style.cursor = 'pointer';
          } else {
            setHoveredChunk(null);
            setTooltip({ show: false, text: '', x: 0, y: 0 });
            document.body.style.cursor = 'default';
          }
        };

        const onClick = async (event) => {
          raycaster.setFromCamera(mouse, camera);
          const intersects = raycaster.intersectObjects(spheres);

          if (intersects.length > 0) {
            const clicked = intersects[0].object;
            try {
              const response = await fetch(
                `http://localhost:5000/api/chunks?file_name=${selectedDocs[0].name}&chunk_index=${clicked.userData.chunkIndex}`
              );
              
              if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
              }
              
              const data = await response.json();
              setSelectedChunkDetails(data.chunk);
              
              // Highlight selected sphere
              spheres.forEach(sphere => {
                if (sphere === clicked) {
                  sphere.material.emissive.setHex(0x888888);
                } else {
                  sphere.material.emissive.setHex(0x000000);
                }
              });
            } catch (error) {
              console.error('Error fetching chunk details:', error);
            }
          }
        };

        renderer.domElement.addEventListener('mousemove', onMouseMove);
        renderer.domElement.addEventListener('click', onClick);

        const animate = () => {
          requestAnimationFrame(animate);
          controls.update();

          // Highlight hovered sphere
          spheres.forEach(sphere => {
            if (sphere === hoveredChunk) {
              sphere.material.emissive.setHex(0x555555);
            } else {
              sphere.material.emissive.setHex(0x000000);
            }
          });

          renderer.render(scene, camera);
        };
        animate();

        return () => {
          renderer.domElement.removeEventListener('mousemove', onMouseMove);
          renderer.domElement.removeEventListener('click', onClick);
        };
      } catch (error) {
        console.error('Error initializing visualization:', error);
      }
    };

    init();

    return () => {
      if (renderer) {
        renderer.dispose();
        containerRef.current?.removeChild(renderer.domElement);
      }
      spheres.forEach(sphere => {
        sphere.geometry.dispose();
        sphere.material.dispose();
      });
    };
  }, [selectedDocs, level]);

  const handleClusterClick = (clusterId) => {
    setSelectedCluster(clusterId);
    setLevel(2);
  };

  return (
    <div className={`panel rag-visualizer ${isExpanded ? 'expanded' : ''}`}>
      <div className="panel-header">
        <h2>RAG Visualizer</h2>
        <div className="visualizer-controls">
          <button onClick={() => setLevel(1)} disabled={level === 1}>
            Document Level
          </button>
          <button onClick={() => setLevel(2)} disabled={level === 2}>
            Chunk Level
          </button>
          <button onClick={onToggleExpand}>
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
      </div>

      <div className="visualization-container" ref={containerRef}>
        {!selectedDocs.length && (
          <div className="placeholder-message">
            Select documents to visualize embeddings
          </div>
        )}
        {tooltip.show && (
          <div 
            className="chunk-tooltip"
            style={{
              position: 'fixed',
              left: tooltip.x + 10,
              top: tooltip.y + 10,
            }}
          >
            {tooltip.text}
          </div>
        )}
      </div>

      {selectedChunkDetails && (
        <div className="chunk-details-panel">
          <h4>Chunk Details</h4>
          <div className="chunk-content">
            <p><strong>Content:</strong></p>
            <p>{selectedChunkDetails.content}</p>
          </div>
          <div className="chunk-keywords">
            <p><strong>Keywords:</strong></p>
            <p>{selectedChunkDetails.keywords.join(', ')}</p>
          </div>
          <button 
            className="close-details"
            onClick={() => setSelectedChunkDetails(null)}
          >
            Close
          </button>
        </div>
      )}

      {visualizationData && (
        <div className="visualization-info">
          <h3>Visualization Info</h3>
          <p>Level: {level === 1 ? 'Document Clusters' : 'Chunk Details'}</p>
          <p>Total Chunks: {visualizationData.reducedData.length}</p>
          {selectedCluster !== null && (
            <div className="cluster-info">
              <h4>Selected Cluster: {selectedCluster}</h4>
              <button onClick={() => setSelectedCluster(null)}>
                Clear Selection
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default RagVisualizer; 