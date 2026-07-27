import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot
} from 'recharts';

export default function RetentionGraph({ dropOffRisks }) {
  const data = useMemo(() => {
    let retention = 100;
    const curve = [];
    const totalPoints = 40; // Higher fidelity for smoother curve

    for (let i = 0; i <= totalPoints; i++) {
      const minutes = (i / totalPoints) * 10;
      
      // Natural decay
      retention = retention - (Math.random() * 1.5 + 0.5);

      const hasDropOff = dropOffRisks.some(risk => {
        const timeStr = risk.timestamp.replace(/\[|\]/g, ''); 
        const [min, sec] = timeStr.split(':').map(Number);
        if (min !== undefined && sec !== undefined) {
          const riskMinutes = min + (sec / 60);
          return Math.abs(minutes - riskMinutes) < 0.25;
        }
        return false;
      });

      if (hasDropOff) {
        retention -= 8; 
      }
      
      if (retention < 0) retention = 0;

      curve.push({
        time: `${Math.floor(minutes)}:${String(Math.floor((minutes % 1) * 60)).padStart(2, '0')}`,
        retention: Math.max(retention, 5), 
        isDropOff: hasDropOff
      });
    }

    return curve;
  }, [dropOffRisks]);

  return (
    <div className="h-[280px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
          <defs>
            <linearGradient id="colorRetention" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ffffff" stopOpacity={0.1}/>
              <stop offset="95%" stopColor="#ffffff" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <XAxis 
            dataKey="time" 
            stroke="#444" 
            tick={{ fill: '#666', fontSize: 11, fontFamily: 'Inter' }}
            tickLine={false}
            axisLine={false}
            minTickGap={40}
            dy={10}
          />
          <YAxis 
            stroke="#444" 
            tick={{ fill: '#666', fontSize: 11, fontFamily: 'Inter' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `${Math.round(value)}%`}
            dx={-10}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#0A0A0A', 
              borderColor: '#222', 
              borderRadius: '8px', 
              color: '#fff',
              boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
              fontFamily: 'Inter',
              fontSize: '12px'
            }}
            itemStyle={{ color: '#fff' }}
            labelStyle={{ color: '#888', marginBottom: '4px' }}
            cursor={{ stroke: '#333', strokeWidth: 1, strokeDasharray: '4 4' }}
          />
          
          <Area 
            type="monotone" 
            dataKey="retention" 
            stroke="#ffffff" 
            strokeWidth={1.5}
            fillOpacity={1} 
            fill="url(#colorRetention)" 
          />

          {data.map((entry, index) => {
            if (entry.isDropOff) {
              return (
                <ReferenceDot 
                  key={index} 
                  x={entry.time} 
                  y={entry.retention} 
                  r={4} 
                  fill="#000" 
                  stroke="#fff" 
                  strokeWidth={1.5} 
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
