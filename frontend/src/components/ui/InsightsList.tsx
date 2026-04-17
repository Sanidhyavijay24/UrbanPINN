"use client";

export const INSIGHTS = [
  {
    title: "Urban Canyon Effect (Wind Tunnels)",
    description: "Identifies streets and corridors where tall buildings channel and predictably accelerate wind, highlighting areas that may spawn uncomfortable or dangerous 'wind tunnels' for pedestrians."
  },
  {
    title: "Stagnation & Pollution Traps",
    description: "Detects 'dead zones' behind large building clusters with very low wind velocities. These areas typically trap vehicle emissions, smog, and summer heat, severely decreasing local air quality."
  },
  {
    title: "Pedestrian Comfort Zones",
    description: "Helps visualize the best locations for outdoor seating, parks, or safe pedestrian crossings by mapping zones that consistently maintain comfortable wind speeds and aerodynamic pressures."
  },
  {
    title: "Urban Heat Islands",
    description: "Exposes how the lack of ventilation and dense concrete placement cause localized thermal hotspots. Pinpoints precise locations requiring shade trees, green roofs, or better structural airflow."
  },
  {
    title: "Architectural Impact",
    description: "Demonstrates how inserting a new high-rise building completely alters the surrounding neighborhood's micro-climate. Empowers architects to test aerodynamic shapes before pouring concrete."
  }
];

interface InsightsListProps {
  /** Start index (inclusive). Defaults to 0. */
  start?: number;
  /** End index (exclusive). Defaults to INSIGHTS.length. */
  end?: number;
}

export function InsightsList({ start = 0, end = INSIGHTS.length }: InsightsListProps) {
  const items = INSIGHTS.slice(start, end);

  return (
    <div className="w-full max-w-xl space-y-1">
      {items.map((insight, index) => (
        <div 
          key={index} 
          className="border-t border-amber-500/20 py-3.5 flex justify-between items-center group cursor-default"
        >
          <div className="space-y-1 pr-6 flex flex-col">
            <span className="text-xs uppercase tracking-[0.15em] font-bold text-amber-400/90 group-hover:text-amber-300 transition-colors">
              {insight.title}
            </span>
            <p className="text-xs text-zinc-500 max-w-md leading-relaxed group-hover:text-zinc-300 transition-colors font-mono">
              {insight.description}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}

