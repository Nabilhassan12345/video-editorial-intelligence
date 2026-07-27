export default function SegmentBreakdown({ segments }) {
  if (!segments || segments.length === 0) {
    return <div className="text-[#888] text-sm italic">No segment data available.</div>;
  }

  return (
    <div className="space-y-4">
      {segments.map((seg, idx) => (
        <div key={idx} className="bg-[#050505] border border-[#222] p-4 rounded-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
          
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-1">
              <span className="font-mono text-xs text-gray-400 bg-[#111] px-2 py-0.5 rounded border border-[#333]">
                {seg.start_time} - {seg.end_time}
              </span>
            </div>
            <p className="text-sm text-gray-200">{seg.text_summary}</p>
          </div>

          <div className="flex items-center space-x-6">
            <div className="text-center">
              <div className="text-[10px] text-[#666] uppercase tracking-wider mb-1">Risk</div>
              <div className={`text-lg font-mono ${seg.risk_score > 6 ? 'text-red-400' : 'text-white'}`}>
                {seg.risk_score}<span className="text-xs text-[#555]">/10</span>
              </div>
            </div>
            <div className="text-center">
              <div className="text-[10px] text-[#666] uppercase tracking-wider mb-1">Engagement</div>
              <div className={`text-lg font-mono ${seg.engagement_score > 7 ? 'text-green-400' : 'text-white'}`}>
                {seg.engagement_score}<span className="text-xs text-[#555]">/10</span>
              </div>
            </div>
          </div>
          
        </div>
      ))}
    </div>
  );
}
