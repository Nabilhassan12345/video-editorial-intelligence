import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
  ReferenceLine
} from 'recharts';

export default function RetentionGraph({ dropOffRisks }) {
  // Generate a fake retention curve that starts at 100% and decays
  // We'll create 20 data points over a hypothetical 10 minute video.
  const data = useMemo(() => {
    let retention = 100;
    const curve = [];
    const totalPoints = 20;

    for (let i = 0; i <= totalPoints; i++) {
      const minutes = (i / totalPoints) * 10;
      
      // Natural decay
      retention = retention - (Math.random() * 3 + 1);

      // Add major drops near the dropOffRisk timestamps
      // Timestamps usually look like "[01:45]"
      const hasDropOff = dropOffRisks.some(risk => {
        const timeStr = risk.timestamp.replace(/\[|\]/g, ''); // "01:45"
        const [min, sec] = timeStr.split(':').map(Number);
        if (min !== undefined && sec !== undefined) {
          const riskMinutes = min + (sec / 60);
          // If this point is close to the risk timestamp, create a big drop
          return Math.abs(minutes - riskMinutes) < 0.5;
        }
        return false;
      });

      if (hasDropOff) {
        retention -= 15; // big drop
      }
      
      if (retention < 0) retention = 0;

      curve.push({
        time: `${Math.floor(minutes)}:${String(Math.floor((minutes % 1) * 60)).padStart(2, '0')}`,
        retention: Math.max(retention, 5), // Keep it above 5% just for visuals
        isDropOff: hasDropOff
      });
    }

    return curve;
  }, [dropOffRisks]);

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="colorRetention" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
          <XAxis 
            dataKey="time" 
            stroke="#9ca3af" 
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            minTickGap={30}
          />
          <YAxis 
            stroke="#9ca3af" 
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `${Math.round(value)}%`}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.8)', borderColor: '#374151', borderRadius: '0.5rem', color: '#fff' }}
            itemStyle={{ color: '#818cf8' }}
            labelStyle={{ color: '#9ca3af', marginBottom: '4px' }}
          />
          
          <Area 
            type="monotone" 
            dataKey="retention" 
            stroke="#6366f1" 
            strokeWidth={3}
            fillOpacity={1} 
            fill="url(#colorRetention)" 
          />

          {/* Highlight Drop-offs */}
          {data.map((entry, index) => {
            if (entry.isDropOff) {
              return (
                <ReferenceDot 
                  key={index} 
                  x={entry.time} 
                  y={entry.retention} 
                  r={5} 
                  fill="#ef4444" 
                  stroke="#111827" 
                  strokeWidth={2} 
                />
              );
            }
            return null;
          })}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
