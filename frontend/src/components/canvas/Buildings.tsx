"use client";

import { useGLTF } from '@react-three/drei';
import { useMemo } from 'react';
import * as THREE from 'three';

export function Buildings() {
  // Graceful degradation when python script hasn't baked the GLTF yet
  let nodes: any = {};
  try {
    const gltf = useGLTF('/models/manhattan_buildings.glb');
    nodes = gltf.nodes;
  } catch (error) {
    console.warn("Building geometry not found. Run python data/scripts/convert_to_gltf.py to bake meshes.");
  }

  // Optimize material for glassmorphism aerodynamic look
  const buildingMaterial = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      color: "#0f172a",       // Deep Slate base
      emissive: "#0284c7",    // Subtle electric blue glow
      emissiveIntensity: 0.3, // Prevents total darkness
      metalness: 0.8,         // Glassy reflection
      roughness: 0.2,         // Smooth finish
      transparent: false,
      opacity: 1.0,           // Fully opaque to block PINN hallucinated internal vectors
    });
  }, []);

  if (!nodes || Object.keys(nodes).length === 0) {
    return null; // Fallback empty
  }

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {Object.entries(nodes).map(([name, node]: [string, any]) => {
        if (node.isMesh) {
          return (
            <mesh 
              key={name}
              geometry={node.geometry}
              material={buildingMaterial}
              castShadow
              receiveShadow
            />
          );
        }
        return null;
      })}
    </group>
  );
}
