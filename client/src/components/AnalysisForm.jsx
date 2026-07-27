import { useState } from 'react';
import { Play, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

export default function AnalysisForm({ onAnalyze, isAnalyzing }) {
  const [transcript, setTranscript] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (transcript.trim()) {
      onAnalyze(transcript);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="premium-panel p-6 md:p-8 relative"
    >
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-3">
          <label htmlFor="transcript" className="block text-xs font-semibold text-[#888888] uppercase tracking-wider">
            Raw Transcript Input
          </label>
          <div className="relative group">
            <textarea
              id="transcript"
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              placeholder="00:00 Today I'm going to show you how I made $10,000 in a single week using a strategy almost no one is talking about..."
              className="premium-input min-h-[160px] resize-y font-mono text-sm leading-relaxed"
              required
              disabled={isAnalyzing}
            />
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={isAnalyzing || !transcript.trim()}
            className="premium-button w-full md:w-auto"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-black" />
                <span>Running Inference...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-black" />
                <span>Analyze Sequence</span>
              </>
            )}
          </button>
        </div>
      </form>
    </motion.div>
  );
}
