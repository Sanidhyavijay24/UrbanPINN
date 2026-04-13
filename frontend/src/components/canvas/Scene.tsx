"use client";

import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei';
import { Buildings } from './Buildings';
import { WindField } from './WindField';
import { ThermalMap } from './ThermalMap';
import { VelocityPoint } from '@/lib/api-client';

interface SceneProps {
  windData: VelocityPoint[];
  sliceHeight: number;
}

export function Scene({ windData, sliceHeight }: SceneProps) {
  return (
    <div className="absolute inset-0 w-full h-full bg-space-950">
      <Canvas
        shadows
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance'
        }}
      >
        <PerspectiveCamera
          makeDefault
          position={[0, 1400, 800]}
          fov={45}
          near={1}
          far={5000}
        />
        
        <OrbitControls
          enablePan={true}
          minDistance={100}
          maxDistance={3000}
          target={[-60, 0, -70]} // Aims exactly at the physical center of the building dataset geometry
          maxPolarAngle={Math.PI / 2 + 0.1} // Allows slightly looking up from ground
        />
        
        {/* Deep, dynamic lighting */}
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[200, 1000, 500]}
          intensity={1.2}
          castShadow
          color="#f1f5f9"
        />
        
        {/* Subtle grid base floor */}
        <Grid
          args={[2000, 2000]}
          position={[0, 0, 0]}
          cellSize={50}
          cellColor="#334155"
          sectionColor="#475569"
          fadeDistance={1500}
          fadeStrength={1}
        />
        
        <Buildings />
        
        {windData.length > 0 && (
          <>
            <ThermalMap data={windData} />
            <WindField data={windData} height={sliceHeight} />
          </>
        )}
        
      </Canvas>
    </div>
  );
}
