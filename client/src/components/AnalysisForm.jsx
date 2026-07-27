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
      className="glass-panel p-6 md:p-8"
    >
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-2">
          <label htmlFor="transcript" className="block text-sm font-medium text-gray-300 ml-1">
            Raw Transcript
          </label>
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
            <textarea
              id="transcript"
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              placeholder="00:00 Today I'm going to show you how I made $10,000 in a single week using a strategy almost no one is talking about..."
              className="glass-input min-h-[200px] resize-y relative z-10 font-mono text-sm leading-relaxed"
              required
              disabled={isAnalyzing}
            />
          </div>
        </div>

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={isAnalyzing || !transcript.trim()}
            className="glass-button flex items-center space-x-2 w-full md:w-auto justify-center"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Analyzing Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                <span>Analyze Video</span>
              </>
            )}
          </button>
        </div>
      </form>
    </motion.div>
  );
}
