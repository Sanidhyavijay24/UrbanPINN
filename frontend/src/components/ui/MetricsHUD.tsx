"use client";

import { VelocityPoint } from '@/lib/api-client';
import { useMemo } from 'react';

interface MetricsHUDProps {
  data: VelocityPoint[];
}

export function MetricsHUD({ data }: MetricsHUDProps) {
  const stats = useMemo(() => {
    if (!data || data.length === 0) return null;
    
    let maxV = 0;
    let sumV = 0;
    let maxP = -Infinity;
    let minP = Infinity;
    
    for (const pt of data) {
      if (pt.magnitude > maxV) maxV = pt.magnitude;
      sumV += pt.magnitude;
      if (pt.p > maxP) maxP = pt.p;
      if (pt.p < minP) minP = pt.p;
    }
    
    return {
      meanV: sumV / data.length,
      maxV,
      pressureGrad: maxP - minP,
    };
  }, [data]);

  if (!stats) return null;

  return (
    <div className="absolute bottom-6 left-6 glass-panel p-4 flex gap-6 text-white z-10 w-auto rounded-xl">
      <div className="flex flex-col">
        <span className="text-xs uppercase text-space-400 font-bold tracking-wider">Mean Velocity</span>
        <span className="text-2xl font-mono text-velocity-40 font-bold">
          {stats.meanV.toFixed(2)} <span className="text-sm text-space-400">m/s</span>
        </span>
      </div>
      <div className="w-px h-full bg-space-700"></div>
      <div className="flex flex-col">
        <span className="text-xs uppercase text-space-400 font-bold tracking-wider">Peak Gusts</span>
        <span className="text-2xl font-mono text-velocity-80 font-bold">
           {stats.maxV.toFixed(2)} <span className="text-sm text-space-400">m/s</span>
        </span>
      </div>
      <div className="w-px h-full bg-space-700"></div>
      <div className="flex flex-col">
        <span className="text-xs uppercase text-space-400 font-bold tracking-wider">Pressure Grad</span>
        <span className="text-2xl font-mono text-rose-400 font-bold">
           {stats.pressureGrad.toFixed(2)} <span className="text-sm text-space-400">Pa</span>
        </span>
      </div>
    </div>
  );
}
