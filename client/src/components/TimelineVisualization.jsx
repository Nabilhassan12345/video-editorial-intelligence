import { AlertTriangle, Flame } from 'lucide-react';

export default function TimelineVisualization({ hook, risks, durationSeconds }) {
  const VIDEO_LENGTH_SECONDS = durationSeconds > 0 ? durationSeconds : 600;
  const formatTime = (secs) =>
    `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(Math.floor(secs % 60)).padStart(2, '0')}`;

  const parseTimestamp = (ts) => {
    // Basic parser for "[01:45]" format
    const match = ts.match(/\[?(\d{1,2}):(\d{2})\]?/);
    if (match) {
      return parseInt(match[1]) * 60 + parseInt(match[2]);
    }
    return 0;
  };

  const hookSeconds = parseTimestamp(hook);
  const hookPercent = Math.min((hookSeconds / VIDEO_LENGTH_SECONDS) * 100, 100);

  return (
    <div className="relative w-full h-12 flex items-center">
      {/* Base timeline track */}
      <div className="absolute left-0 right-0 h-1.5 bg-[#222] rounded-full"></div>
      
      {/* Hook Marker */}
      <div 
        className="absolute w-4 h-4 bg-white rounded-full flex items-center justify-center shadow-[0_0_10px_rgba(255,255,255,0.5)] z-10 -translate-y-1/2 top-1/2"
        style={{ left: `${hookPercent}%`, transform: 'translate(-50%, -50%)' }}
      >
        <Flame className="w-2 h-2 text-black" />
      </div>

      {/* Risk Markers */}
      {risks.map((risk, idx) => {
        const sec = parseTimestamp(risk.timestamp);
        const percent = Math.min((sec / VIDEO_LENGTH_SECONDS) * 100, 100);
        return (
          <div 
            key={idx}
            className="absolute w-3 h-3 bg-red-500 rounded-full flex items-center justify-center shadow-[0_0_10px_rgba(239,68,68,0.5)] z-10 -translate-y-1/2 top-1/2 cursor-pointer group"
            style={{ left: `${percent}%`, transform: 'translate(-50%, -50%)' }}
          >
            {/* Tooltip */}
            <div className="absolute bottom-6 opacity-0 group-hover:opacity-100 transition-opacity bg-[#111] border border-[#333] text-xs p-2 rounded w-48 text-left shadow-lg pointer-events-none z-20">
              <div className="text-red-400 font-bold mb-1">{risk.timestamp} Drop-off</div>
              <div className="text-gray-300">{risk.reason}</div>
            </div>
          </div>
        );
      })}

      {/* Labels */}
      <div className="absolute top-8 left-0 text-[10px] text-[#666] font-mono">00:00</div>
      <div className="absolute top-8 right-0 text-[10px] text-[#666] font-mono">{formatTime(VIDEO_LENGTH_SECONDS)}</div>
    </div>
  );
}
