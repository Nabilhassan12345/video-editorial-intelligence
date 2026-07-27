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

  const handleLoadDemo = () => {
    const demoScript = `[00:00] Today I'm going to show you how I made $10,000 in a single week using a strategy almost no one is talking about.
[00:15] Most people think you need to grind 80 hours a week to see these kinds of returns, but they're completely wrong.
[00:45] Let's dive into the exact framework I used. It's broken down into three simple steps.
[01:45] Step one involves leveraging asymmetric risk algorithms. This requires a deep understanding of standard deviation and market sentiment divergence. 
[03:20] Now, if we look at the historical data, you can clearly see the pattern emerging right here on the chart.
[04:20] *Long pause as screen records the chart loading slowly...*
[05:10] So, anyway, moving on to step two...
[08:10] We use the RSI divergence multiplier coefficient to extract the delta. It's essentially a proprietary metric that requires a $500/month subscription to calculate accurately.
[09:30] And that's exactly how it's done. Make sure to like and subscribe if you found this helpful.`;
    setTranscript(demoScript);
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
                  <button 
                    type="button" 
                    onClick={handleLoadDemo}
                    className="text-xs text-blue-400 hover:text-blue-300 font-medium"
                  >
                    Load Demo Script
                  </button>
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
