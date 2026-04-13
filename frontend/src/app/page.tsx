"use client";

import { useState, useEffect, useMemo } from 'react';
import { Scene } from '@/components/canvas/Scene';
import { VelocityPoint } from '@/lib/api-client';
import { MetricsHUD } from '@/components/ui/MetricsHUD';

export default function SimulationPage() {
  const [sliceHeight, setSliceHeight] = useState(50); // 50m default
  const [windData, setWindData] = useState<VelocityPoint[]>([]);
  const [loading, setLoading] = useState(false);
  
  const { peakGust, minTemp, maxTemp } = useMemo(() => {
    if (!windData || windData.length === 0) return { peakGust: 0, minTemp: 0, maxTemp: 0 };
    let tempMaxV = 0;
    let tempMinT = Infinity;
    let tempMaxT = -Infinity;
    
    for (const p of windData) {
      if (p.magnitude > tempMaxV) tempMaxV = p.magnitude;
      if (p.t < tempMinT) tempMinT = p.t;
      if (p.t > tempMaxT) tempMaxT = p.t;
    }
    
    return { peakGust: tempMaxV, minTemp: tempMinT, maxTemp: tempMaxT };
  }, [windData]);
  
  // Basic naive fetcher before we implement robust React Query next
  useEffect(() => {
    async function fetchSlice() {
      setLoading(true);
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/api/v1/predict/slice`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ z_height: sliceHeight, grid_resolution: 80 })
        });
        
        if (response.ok) {
          const data = await response.json();
          setWindData(data.data);
        }
      } catch (err) {
        console.error("Failed to fetch wind slice.", err);
      } finally {
        setLoading(false);
      }
    }
    
    fetchSlice();
  }, [sliceHeight]);

  return (
    <main className="relative w-screen h-screen overflow-hidden bg-space-950">
      <Scene windData={windData} sliceHeight={sliceHeight} />
      
      {/* Temporary basic HUD layout for testing */}
      <div className="absolute top-6 left-6 glass-elevated p-6 w-80 z-10 flex flex-col gap-6 text-white">
        <div>
          <h1 className="text-xl font-bold font-sans tracking-wide">URBAN PINN</h1>
          <p className="text-sm text-space-400">Micro-climate CFD Simulator</p>
        </div>
        
        <div className="flex flex-col gap-2">
          <label className="text-xs uppercase tracking-widest text-space-300 font-bold">
            Slice Height: <span className="text-velocity-60 text-lg ml-2">{sliceHeight}m</span>
          </label>
            <input 
              type="range" 
              min="0" 
              max="120" 
              step="5"
              value={sliceHeight}
              onChange={(e) => setSliceHeight(parseInt(e.target.value))}
              className="w-full h-2 bg-space-800 rounded-lg appearance-none cursor-pointer"
            />
            
            {/* Velocity Spectrum Legend */}
            <div className="pt-5 mt-2 border-t border-space-800/80">
              <div className="flex justify-between items-center text-[11px] font-bold text-space-400 uppercase tracking-widest mb-2">
                <span>Velocity Spectrum</span>
                <span>Peak: {peakGust.toFixed(1)} m/s</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gradient-to-r from-sky-500 via-emerald-400 via-yellow-400 to-red-500 shadow-[0_0_10px_rgba(16,185,129,0.2)]"></div>
              <div className="flex justify-between text-[10px] text-space-500 font-mono mt-2">
                <span>0.0</span>
                <span>Local Max</span>
              </div>
            </div>

            {/* Thermal Temperature Legend */}
            <div className="pt-4 mt-2">
              <div className="flex justify-between items-center text-[11px] font-bold text-space-400 uppercase tracking-widest mb-2">
                <span>Ambient Thermal</span>
                <span className="text-red-400">Max: {maxTemp !== -Infinity ? maxTemp.toFixed(1) : "0.0"}°</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gradient-to-r from-blue-600 via-purple-500 via-red-500 via-yellow-400 to-white shadow-[0_0_10px_rgba(138,43,226,0.2)]"></div>
              <div className="flex justify-between text-[10px] text-space-500 font-mono mt-2">
                <span className="text-blue-400">{minTemp !== Infinity ? minTemp.toFixed(1) : "0.0"}°</span>
                <span>Hot</span>
              </div>
            </div>
          </div>
        
        {loading && (
          <div className="text-sm text-velocity-40 animate-pulse">Running Neural Inference...</div>
        )}
      </div>

      <MetricsHUD data={windData} />
    </main>
  );
}
