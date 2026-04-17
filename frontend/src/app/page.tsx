"use client";

import { useState, useEffect, useMemo } from 'react';
import { Scene } from '@/components/canvas/Scene';
import { VelocityPoint } from '@/lib/api-client';
import { MetricsHUD } from '@/components/ui/MetricsHUD';
import { InsightsList } from '@/components/ui/InsightsList';

export default function SimulationPage() {
  const [sliceHeight, setSliceHeight] = useState(75); // 75m default
  const [windData, setWindData] = useState<VelocityPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

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

  // Debounced fetcher to prevent overwhelming the remote Cloud CPU
  useEffect(() => {
    let timeoutId: NodeJS.Timeout;

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

    timeoutId = setTimeout(() => {
      fetchSlice();
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [sliceHeight]);

  return (
    <main className="relative min-h-[100dvh] w-screen overflow-x-hidden pt-6 md:pt-10 selection:bg-amber-500/30 font-body">
      {/* 
        ========================================
        LEFT COLUMN: HERO TEXT & INSIGHTS 
        ========================================
      */}
      <div className={`fixed top-2 md:top-4 left-0 pl-6 md:pl-10 lg:pl-12 pr-8 w-full md:w-[50%] lg:w-[45%] transition-opacity duration-700 ${isFullscreen ? 'opacity-60 pointer-events-none blur-sm' : 'opacity-100'}`}>
        <div className="flex flex-col items-start space-y-6">
          <div className="relative">
            {/* Ambient glow */}
            <div className="absolute -top-16 -left-16 w-72 h-72 bg-amber-500/5 rounded-full blur-[120px] pointer-events-none"></div>
            <h1 className="text-6xl md:text-7xl lg:text-8xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-br from-white to-space-500 pb-2 relative">
              URBAN<br />PINN.
            </h1>
            <p className="text-[10px] md:text-xs uppercase tracking-[0.3em] text-amber-500/60 font-mono mt-2 relative">Micro-Climate CFD Simulator</p>
          </div>
          <InsightsList start={0} end={4} />
        </div>
      </div>

      {/* 
        ========================================
        RIGHT COLUMN / FULLSCREEN SIMULATION 
        ========================================
      */}
      <div
        className={`fixed overflow-hidden transition-all duration-700 ease-[cubic-bezier(0.25,1,0.5,1)] ${isFullscreen
            ? 'inset-4 md:inset-12 lg:inset-16 z-50 rounded-2xl bg-space-950/90 backdrop-blur-3xl border border-white/10 shadow-[0_0_100px_rgba(0,0,0,1)]'
            : 'top-8 right-8 w-full max-w-sm md:max-w-md lg:max-w-[45%] aspect-[4/3] rounded-xl bg-space-950/40 backdrop-blur-md hover:border-amber-400/50 shadow-2xl z-20 group border border-white/10'
          }`}
      >
        <Scene windData={windData} sliceHeight={sliceHeight} />

        {/* If Not Fullscreen: Reading Mode Mini-HUD Overlay */}
        {!isFullscreen && (
          <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-5 z-30 bg-gradient-to-t from-space-950/80 via-transparent to-space-950/30">
            <div className="flex justify-between items-start opacity-70 group-hover:opacity-100 transition-opacity">
              <div className="space-y-1">
                <p className="text-[10px] uppercase tracking-[0.2em] text-amber-500 font-bold">Simulation Active</p>
                <h3 className="text-white font-bold text-sm">Manhattan Grid Inference</h3>
              </div>
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-amber-500"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.29 7 12 12 20.71 7"></polyline><line x1="12" y1="22" x2="12" y2="12"></line></svg>
            </div>

            <div className="flex justify-center opacity-0 group-hover:opacity-100 transition-opacity absolute top-12 left-0 right-0 z-40 pointer-events-none">
              <button
                onClick={() => setIsFullscreen(true)}
                className="bg-amber-500/20 text-amber-500 hover:bg-amber-500/40 hover:text-white transition-colors border border-amber-500/50 px-4 py-2 rounded-full text-xs font-bold uppercase tracking-widest backdrop-blur-md pointer-events-auto shadow-lg"
              >
                Click to Explore Fullscreen
              </button>
            </div>

            <div className="space-y-2 opacity-70 group-hover:opacity-100 transition-opacity">
              <div className="flex justify-between text-[9px] text-gray-300 font-mono">
                <span>LAT: 40.7128° N</span>
                <span>LONG: 74.0060° W</span>
              </div>
            </div>
          </div>
        )}

        {/* If Fullscreen: The Full Interactive Controls */}
        <div className={`absolute top-6 left-6 transition-all duration-700 delay-100 ${isFullscreen ? 'opacity-100 translate-y-0 pointer-events-auto' : 'opacity-0 -translate-y-4 pointer-events-none'}`}>
          <div className="glass-elevated p-6 w-80 flex flex-col gap-6 text-white border-white/5">
            <div className="flex justify-between items-center bg-black/20 -m-6 mb-2 p-6 border-b border-white/10 rounded-t-2xl">
              <div>
                <h1 className="text-xl font-bold font-headline tracking-wide">URBAN PINN</h1>
                <p className="text-sm text-space-400">Micro-climate CFD Simulator</p>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); setIsFullscreen(false); }}
                className="bg-white/5 hover:bg-white/15 p-2.5 rounded-full transition-colors backdrop-blur-md border border-white/10"
                title="Minimize Simulation"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path></svg>
              </button>
            </div>

            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-space-300 font-bold">
                Slice Height: <span className="text-amber-400 text-lg ml-2">{sliceHeight}m</span>
              </label>
              <input
                type="range"
                min="0"
                max="120"
                step="5"
                value={sliceHeight}
                onChange={(e) => setSliceHeight(parseInt(e.target.value))}
                onClick={(e) => e.stopPropagation()}
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
          </div>
        </div>

        {/* Metrics HUD - visible only in fullscreen */}
        {isFullscreen && (
          <div className="absolute bottom-4 right-6 z-[60]">
            <MetricsHUD data={windData} />
          </div>
        )}

        {/* Loading Indicator */}
        {loading && isFullscreen && (
          <div className="absolute bottom-6 left-6 text-sm text-amber-500 animate-pulse bg-space-950/80 px-4 py-2 rounded-full border border-amber-500/20 z-50 backdrop-blur-md">
            Running Neural Inference...
          </div>
        )}
      </div>

      {/* 5th Insight — Positioned below the simulation window on the right */}
      <div className={`fixed top-[calc(8px+75vh+32px)] right-8 w-full max-w-sm md:max-w-md lg:max-w-[45%] z-10 transition-opacity duration-700 ${isFullscreen ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
        <InsightsList start={4} end={5} />
      </div>

    </main>
  );
}
