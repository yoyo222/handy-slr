import React from 'react';
import { pinwheel } from 'ldrs';

pinwheel.register();

const LoadingScreen: React.FC = () => {
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundColor: 'black', 
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 9999
    }}>
      <l-pinwheel
        size="100" 
        stroke="4"
        speed="0.9"
        color="white"
      ></l-pinwheel>
    </div>
  );
};

export default LoadingScreen;