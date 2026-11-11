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

const toSeconds = (ts) => {
  const match = String(ts || '').match(/(\d{1,2}):(\d{2})/);
  return match ? parseInt(match[1], 10) * 60 + parseInt(match[2], 10) : null;
};

const formatTime = (secs) =>
  `${Math.floor(secs / 60)}:${String(Math.floor(secs % 60)).padStart(2, '0')}`;

/**
 * Plots a heuristic retention index (100 = start). Uses the Python pipeline's
 * modeled_retention_index when available; otherwise derives the same kind of
 * index deterministically from per-segment risk scores. Not measured retention.
 */
export default function RetentionGraph({ retentionCurve, segments = [], dropOffRisks = [] }) {
  const data = useMemo(() => {
    const riskSeconds = dropOffRisks.map(r => toSeconds(r.timestamp)).filter(s => s !== null);
    const isRisk = (secs) => riskSeconds.some(r => Math.abs(r - secs) < 1);

    if (Array.isArray(retentionCurve) && retentionCurve.length > 0) {
      return retentionCurve.map(pt => ({
        time: formatTime(pt.seconds),
        index: pt.index,
        isDropOff: isRisk(pt.seconds),
      }));
    }

    let index = 100;
    return segments.map(seg => {
      const secs = toSeconds(seg.start_time) ?? 0;
      index = Math.max(index - (seg.risk_score || 0) * 0.8, 0);
      return { time: formatTime(secs), index: Math.round(index * 10) / 10, isDropOff: isRisk(secs) };
    });
  }, [retentionCurve, segments, dropOffRisks]);

  if (data.length === 0) {
    return <div className="text-xs text-[#666]">No segment data available to plot.</div>;
  }

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
            domain={[0, 100]}
            tickFormatter={(value) => `${Math.round(value)}`}
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
            formatter={(value) => [value, 'Heuristic index']}
          />

          <Area
            type="monotone"
            dataKey="index"
            stroke="#ffffff"
            strokeWidth={1.5}
            fillOpacity={1}
            fill="url(#colorRetention)"
          />

          {data.map((entry, idx) => entry.isDropOff ? (
            <ReferenceDot
              key={idx}
              x={entry.time}
              y={entry.index}
              r={4}
              fill="#000"
              stroke="#fff"
              strokeWidth={1.5}
            />
          ) : null)}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
