import { useState, useEffect, useRef } from 'react';
import { Terminal, Loader2 } from 'lucide-react';

// Illustrative progress steps shown while the request is in flight; they are not live server logs.
const LOG_SEQUENCE = [
  { text: "Submitting transcript to local Node.js API...", delay: 200 },
  { text: "Segmenting transcript by timestamp...", delay: 700 },
  { text: "Computing lexical & pacing features (WPM, filler ratio)...", delay: 1300 },
  { text: "Running DistilBERT + LoRA multi-head model on CPU (if checkpoint available)...", delay: 1900 },
  { text: "Falling back to OpenAI / mock response if the Python pipeline is unavailable...", delay: 2600 },
  { text: "Building evidence-grounded drop-off explanations...", delay: 3300 },
  { text: "Formatting output JSON...", delay: 4000 },
  { text: "Waiting for server response...", delay: 4500 }
];

export default function InferenceTerminal() {
  const [logs, setLogs] = useState([]);
  const bottomRef = useRef(null);

  useEffect(() => {
    const timeouts = LOG_SEQUENCE.map((logItem, index) => {
      return setTimeout(() => {
        setLogs(prev => {
          // Check if already includes to prevent double render issues in strict mode
          if (prev.some(l => l.text === logItem.text)) return prev;
          return [...prev, { ...logItem, timestamp: (logItem.delay / 1000).toFixed(2) }];
        });
      }, logItem.delay);
    });

    return () => timeouts.forEach(clearTimeout);
  }, []);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  return (
    <div className="w-full h-[320px] bg-[#050505] rounded-xl border border-[#222] shadow-[0_0_40px_rgba(0,0,0,0.8)] overflow-hidden flex flex-col font-mono relative">
      {/* Fake window controls */}
      <div className="h-10 bg-[#0a0a0a] border-b border-[#222] flex items-center px-4 justify-between select-none">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500/20 border border-yellow-500/50"></div>
          <div className="w-3 h-3 rounded-full bg-green-500/20 border border-green-500/50"></div>
        </div>
        <div className="flex items-center space-x-2 text-[#555] text-xs">
          <Terminal className="w-3 h-3" />
          <span>local pipeline (illustrative steps)</span>
        </div>
        <div className="w-16"></div> {/* Spacer */}
      </div>

      <div className="flex-1 p-4 overflow-y-auto text-xs leading-relaxed text-[#00ffcc]">
        <div className="text-[#666] mb-4">
          $ POST /analyze (transcript)
        </div>
        
        {logs.map((log, i) => (
          <div key={i} className="mb-2 flex items-start space-x-3">
            <span className="text-[#555] shrink-0">[{log.timestamp}s]</span>
            <span className={log.text.includes('WARNING') ? 'text-yellow-400' : 'text-[#00ffcc]'}>
              {log.text}
            </span>
          </div>
        ))}
        
        <div className="flex items-center space-x-2 mt-4 text-[#555]">
          <Loader2 className="w-3 h-3 animate-spin" />
          <span>Processing...</span>
        </div>
        
        <div ref={bottomRef} />
      </div>
      
      {/* CRT Scanline overlay effect */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.03] mix-blend-overlay" style={{
        backgroundImage: 'linear-gradient(rgba(255, 255, 255, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))',
        backgroundSize: '100% 4px, 6px 100%'
      }}></div>
    </div>
  );
}
