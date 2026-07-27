import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FileVideo } from 'lucide-react';
import AnalysisForm from '../components/AnalysisForm';

export default function Analyze() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleAnalyze = async (transcript) => {
    setIsAnalyzing(true);
    setError('');

    try {
      const response = await fetch('http://localhost:3001/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript })
      });

      if (!response.ok) {
        throw new Error('Failed to analyze transcript');
      }

      const data = await response.json();
      // Redirect to the new results page with the ID
      navigate(`/results/${data.id}`);
    } catch (err) {
      setError(err.message || 'An error occurred during analysis.');
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">
      <header className="space-y-2">
        <div className="flex items-center space-x-3 text-[#888] mb-2">
          <FileVideo className="w-5 h-5" />
          <span className="text-sm font-semibold uppercase tracking-wider">New Analysis Task</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-white">
          Transcript Intelligence
        </h1>
        <p className="text-[#888] max-w-2xl font-light">
          Paste your raw video transcript below. Our ML pipeline will segment the text, predict retention decay, and output actionable editorial insights.
        </p>
      </header>

      <main>
        <AnalysisForm onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />

        {error && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-6 p-4 bg-red-950/30 border border-red-900/50 rounded-lg text-red-400 text-sm"
          >
            {error}
          </motion.div>
        )}
      </main>
    </div>
  );
}
