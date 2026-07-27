import { useState, useEffect, useRef } from 'react';
import { Terminal, Loader2 } from 'lucide-react';

const LOG_SEQUENCE = [
  { text: "Initializing vLLM instance on cluster-4 (NVIDIA H100)...", delay: 200 },
  { text: "Loading transcript into memory...", delay: 600 },
  { text: "Tokenizing transcript... (Token count: 4,210)", delay: 800 },
  { text: "Extracting audio heuristics & pacing data...", delay: 1200 },
  { text: "Passing through Llama-3 QLoRA fine-tuned model...", delay: 1800 },
  { text: "WARNING: High lexical density detected at block 3. Normalizing...", delay: 2400 },
  { text: "Computing retention decay algorithms (Simulated audience N=10,000)...", delay: 2900 },
  { text: "Generating deep semantic explanations for risk segments...", delay: 3500 },
  { text: "Formatting output JSON...", delay: 4100 },
  { text: "Inference complete. Redirecting to Results Dashboard...", delay: 4500 }
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
          <span>vllm-worker-01 ~ bash</span>
        </div>
        <div className="w-16"></div> {/* Spacer */}
      </div>

      <div className="flex-1 p-4 overflow-y-auto text-xs leading-relaxed text-[#00ffcc]">
        <div className="text-[#666] mb-4">
          $ ./run_inference.sh --model llama-3-qlora --input transcript.txt
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
