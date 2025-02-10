import React, { useState } from 'react';
import '../styles/Sidebar.css';

function Sidebar() {
  const [instances, setInstances] = useState([]);
  const [editingId, setEditingId] = useState(null);

  const createNewInstance = () => {
    const newInstance = {
      id: Date.now(),
      name: `Instance ${instances.length + 1}`,
    };
    setInstances([...instances, newInstance]);
  };

  const handleNameEdit = (id, newName) => {
    setInstances(instances.map(instance => 
      instance.id === id ? { ...instance, name: newName } : instance
    ));
    setEditingId(null);
  };

  return (
    <div className="sidebar">
      <div className="uploaded-docs">
        <h2>Uploaded Docs</h2>
        {/* Document list will be populated here */}
      </div>

      <button className="new-instance-btn" onClick={createNewInstance}>
        New Instance
      </button>

      <div className="instances-list">
        {instances.map(instance => (
          <div key={instance.id} className="instance-item">
            {editingId === instance.id ? (
              <input
                type="text"
                value={instance.name}
                onChange={(e) => handleNameEdit(instance.id, e.target.value)}
                onBlur={() => setEditingId(null)}
                autoFocus
              />
            ) : (
              <div 
                className="instance-name"
                onClick={() => setEditingId(instance.id)}
              >
                {instance.name}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Sidebar; 