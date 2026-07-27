import { useState } from 'react';
import { Play } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import InferenceTerminal from './InferenceTerminal';

export default function AnalysisForm({ onAnalyze, isAnalyzing }) {
  const [transcript, setTranscript] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (transcript.trim()) {
      onAnalyze(transcript);
    }
  };

  return (
    <div className="relative">
      <AnimatePresence mode="wait">
        {isAnalyzing ? (
          <motion.div
            key="terminal"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
          >
            <InferenceTerminal />
          </motion.div>
        ) : (
          <motion.div 
            key="form"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="premium-panel p-6 md:p-8"
          >
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label htmlFor="transcript" className="block text-xs font-semibold text-[#888888] uppercase tracking-wider">
                    Raw Transcript Input
                  </label>
                </div>
                <div className="relative group">
                  <textarea
                    id="transcript"
                    value={transcript}
                    onChange={(e) => setTranscript(e.target.value)}
                    placeholder="00:00 Today I'm going to show you how I made $10,000 in a single week using a strategy almost no one is talking about..."
                    className="premium-input min-h-[220px] resize-y font-mono text-sm leading-relaxed"
                    required
                  />
                </div>
              </div>

              <div className="flex justify-end pt-2">
                <button
                  type="submit"
                  disabled={!transcript.trim()}
                  className="premium-button w-full md:w-auto"
                >
                  <Play className="w-4 h-4 fill-black" />
                  <span>Analyze Sequence</span>
                </button>
              </div>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
