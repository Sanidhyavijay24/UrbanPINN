"use client";

import { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { VelocityPoint } from '@/lib/api-client';

interface WindFieldProps {
  data: VelocityPoint[];
  height: number;
}

export function WindField({ data, height }: WindFieldProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  useEffect(() => {
    if (!meshRef.current || !data) return;

    let maxMag = 1.0;
    for (const p of data) {
      if (p.magnitude > maxMag) maxMag = p.magnitude;
    }

    const dummy = new THREE.Object3D();
    
    data.forEach((point, i) => {
      // Map domain coordinates properly
      dummy.position.set(point.x, point.z, -point.y);
      
      // Filter out only absolute dead zones (0.05) to preserve trapped wake air in narrow alleys
      if (point.magnitude < 0.05) {
        dummy.scale.set(0, 0, 0);
        dummy.updateMatrix();
        meshRef.current!.setMatrixAt(i, dummy.matrix);
        return;
      }

      // Scale to make them look like sleek dynamic darts
      const baseScale = Math.max(point.magnitude / 1.5, 1.5); 
      dummy.scale.set(baseScale * 1.2, baseScale * 2.5, baseScale * 1.2);
      
      // Orient the dart
      const direction = new THREE.Vector3(point.u, point.w, -point.v).normalize();
      dummy.lookAt(dummy.position.clone().add(direction));
      // Cone geometry points up (+Y axis). In three.js lookAt points the +Z axis.
      // So we rotate the dummy 90 degrees on X to point the cone tip forward toward +Z direction.
      dummy.rotateX(Math.PI / 2);
      
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
      
      // Dynamic color interpolation based on Local Max Wind
      const vColor = new THREE.Color()
      const t = Math.max(0, Math.min(1, point.magnitude / maxMag))
      
      if (t < 0.2) vColor.lerpColors(new THREE.Color(0x0284c7), new THREE.Color(0x06b6d4), t * 5.0); // Blue to Cyan
      else if (t < 0.4) vColor.lerpColors(new THREE.Color(0x06b6d4), new THREE.Color(0x10b981), (t - 0.2) * 5.0); // Cyan to Emerald
      else if (t < 0.6) vColor.lerpColors(new THREE.Color(0x10b981), new THREE.Color(0xeab308), (t - 0.4) * 5.0); // Emerald to Yellow
      else if (t < 0.8) vColor.lerpColors(new THREE.Color(0xeab308), new THREE.Color(0xf97316), (t - 0.6) * 5.0); // Yellow to Orange
      else vColor.lerpColors(new THREE.Color(0xf97316), new THREE.Color(0xef4444), (t - 0.8) * 5.0); // Orange to Red
      
      meshRef.current!.setColorAt(i, vColor);
    });
    
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  }, [data]);

  if (!data || data.length === 0) return null;

  return (
    <instancedMesh ref={meshRef} args={[null as any, null as any, data.length]} frustumCulled>
      <coneGeometry args={[0.8, 5, 4]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}
