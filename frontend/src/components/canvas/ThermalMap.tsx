"use client";

import { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { VelocityPoint } from '@/lib/api-client';

interface ThermalMapProps {
  data: VelocityPoint[];
}

export function ThermalMap({ data }: ThermalMapProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  useEffect(() => {
    if (!meshRef.current || !data || data.length === 0) return;

    // We assume the NN outputs T roughly normalized or we find its min/max dynamically
    let minT = Infinity;
    let maxT = -Infinity;
    for (const p of data) {
      if (p.t < minT) minT = p.t;
      if (p.t > maxT) maxT = p.t;
    }

    const tRange = maxT - minT || 1.0;

    const dummy = new THREE.Object3D();
    
    data.forEach((point, i) => {
      // Map domain coordinates properly
      dummy.position.set(point.x, point.z, -point.y);
      
      // Lay them completely flat along the XZ plane to form a tile subgrid
      dummy.rotation.set(-Math.PI / 2, 0, 0); 
      
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
      
      // Thermal Gradient mapping (Blue -> Purple -> Red -> Yellow -> White)
      const vColor = new THREE.Color()
      const normalizedT = Math.max(0, Math.min(1, (point.t - minT) / tRange))
      
      if (normalizedT < 0.25) vColor.lerpColors(new THREE.Color(0x0000ff), new THREE.Color(0x8a2be2), normalizedT * 4.0); // Blue to Purple
      else if (normalizedT < 0.5) vColor.lerpColors(new THREE.Color(0x8a2be2), new THREE.Color(0xff0000), (normalizedT - 0.25) * 4.0); // Purple to Red
      else if (normalizedT < 0.75) vColor.lerpColors(new THREE.Color(0xff0000), new THREE.Color(0xffff00), (normalizedT - 0.5) * 4.0); // Red to Yellow
      else vColor.lerpColors(new THREE.Color(0xffff00), new THREE.Color(0xffffff), (normalizedT - 0.75) * 4.0); // Yellow to White
      
      meshRef.current!.setColorAt(i, vColor);
    });
    
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  }, [data]);

  if (!data || data.length === 0) return null;

  return (
    <instancedMesh ref={meshRef} args={[null as any, null as any, data.length]} frustumCulled>
      <planeGeometry args={[15, 16]} />
      {/* Basic material to ensure colors pop, sheer opacity so buildings can be seen below */}
      <meshBasicMaterial toneMapped={false} transparent opacity={0.35} depthWrite={false} side={THREE.DoubleSide} />
    </instancedMesh>
  );
}
