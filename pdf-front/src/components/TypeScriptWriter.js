import React, { useState, useEffect, useRef } from 'react';

const TypewriterResponse = ({ text, speed = 20 }) => {
  // Use localStorage to persist the current state
  const [displayedText, setDisplayedText] = useState(() => {
    // Try to retrieve previous state from localStorage for this specific text
    const saved = localStorage.getItem(`typewriter-${text.slice(0, 20)}`);
    return saved || '';
  });
  
  const [isComplete, setIsComplete] = useState(() => {
    return localStorage.getItem(`typewriter-complete-${text.slice(0, 20)}`) === 'true';
  });
  
  const [currentIndex, setCurrentIndex] = useState(() => {
    const saved = localStorage.getItem(`typewriter-index-${text.slice(0, 20)}`);
    return saved ? parseInt(saved, 10) : 0;
  });
  
  const containerRef = useRef(null);
  const textKey = useRef(text.slice(0, 20)); // Create a reference to identify this text

  // Reset when text changes
  useEffect(() => {
    if (text.slice(0, 20) !== textKey.current) {
      setDisplayedText('');
      setCurrentIndex(0);
      setIsComplete(false);
      textKey.current = text.slice(0, 20);
    }
  }, [text]);

  // Typewriter effect
  useEffect(() => {
    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        const newDisplayedText = displayedText + text[currentIndex];
        setDisplayedText(newDisplayedText);
        setCurrentIndex(prev => prev + 1);
        
        // Save state to localStorage
        localStorage.setItem(`typewriter-${textKey.current}`, newDisplayedText);
        localStorage.setItem(`typewriter-index-${textKey.current}`, currentIndex + 1);
        
        if (containerRef.current) {
          containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
      }, speed);
      
      return () => clearTimeout(timer);
    } else if (currentIndex === text.length && !isComplete) {
      setIsComplete(true);
      localStorage.setItem(`typewriter-complete-${textKey.current}`, 'true');
    }
  }, [text, currentIndex, displayedText, speed, isComplete]);

  // Split text into paragraphs for better formatting
  const paragraphs = displayedText.split('\n').filter(p => p.trim() !== '');

  return (
    <div 
      ref={containerRef}
      className="typewriter-container"
      style={{
        width: '100%',
        lineHeight: '1.5',
      }}
    >
      {paragraphs.map((paragraph, idx) => (
        <p key={idx} style={{ marginBottom: '0.8rem' }}>
          {paragraph}
        </p>
      ))}
      {!isComplete && (
        <span className="cursor" style={{ 
          display: 'inline-block',
          width: '0.5rem',
          height: '1.2rem',
          backgroundColor: '#333',
          marginLeft: '2px',
          animation: 'blink 1s step-end infinite',
        }}></span>
      )}
      <style jsx>{`
        @keyframes blink {
          from, to { opacity: 1; }
          50% { opacity: 0; }
        }
      `}</style>
    </div>
  );
};

export default TypewriterResponse;