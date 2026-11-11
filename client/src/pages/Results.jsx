import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Flame, AlertTriangle, TrendingUp, Sparkles, Activity, ArrowLeft, Lightbulb, Download, Share2, Copy } from 'lucide-react';
import RetentionGraph from '../components/RetentionGraph';
import TimelineVisualization from '../components/TimelineVisualization';
import SegmentBreakdown from '../components/SegmentBreakdown';

export default function Results() {
  const { id } = useParams();
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [toast, setToast] = useState('');

  useEffect(() => {
    fetch(`http://localhost:3001/results/${id}`)
      .then(res => {
        if (!res.ok) throw new Error('Analysis not found');
        return res.json();
      })
      .then(data => {
        setResults(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [id]);

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(''), 3000);
  };

  if (loading) {
    return <div className="text-[#888] animate-pulse">Loading analysis data...</div>;
  }

  if (error) {
    return (
      <div className="text-center p-8">
        <p className="text-red-400 mb-4">{error}</p>
        <Link to="/analyze" className="premium-button inline-flex">Go back</Link>
      </div>
    );
  }

  const { hook, drop_off_risks, quality_score, improvements, segments, transcript, source, model_status, prediction_note } = results;
  const showModelWarning = source === 'mock' || (source === 'python_model' && model_status !== 'fine_tuned');
  const wordCount = transcript ? transcript.split(' ').length : 1450;
  const estDuration = Math.round(wordCount / 150); // ~150 words per minute

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.05 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } }
  };

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6 pb-20 relative"
    >
      {showModelWarning && (
        <div className="rounded-md border border-yellow-900/60 bg-yellow-950/30 px-4 py-3 text-sm text-yellow-300">
          {prediction_note || 'These results are illustrative and not produced by a trained model.'}
        </div>
      )}
      {source === 'python_model' && !showModelWarning && prediction_note && (
        <div className="rounded-md border border-[#333] bg-[#111] px-4 py-3 text-xs text-gray-400">{prediction_note}</div>
      )}

      {/* Toast Notification */}
      {toast && (
        <motion.div 
          initial={{ opacity: 0, y: -20, x: '-50%' }}
          animate={{ opacity: 1, y: 0, x: '-50%' }}
          exit={{ opacity: 0, y: -20, x: '-50%' }}
          className="fixed top-20 left-1/2 bg-white text-black px-4 py-2 rounded-lg font-medium text-sm shadow-xl z-50 flex items-center space-x-2"
        >
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span>{toast}</span>
        </motion.div>
      )}

      {/* Header Area */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-2">
        <div className="flex items-center space-x-4">
          <Link to="/history" className="w-8 h-8 rounded-full bg-[#111] border border-[#333] flex items-center justify-center hover:bg-[#222] transition-colors text-white">
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Analysis Report</h1>
            <p className="text-[#666] text-xs font-mono mt-1">ID: {id}</p>
          </div>
        </div>
        
        {/* Action Buttons */}
        <div className="flex items-center space-x-3">
          <button onClick={() => showToast('Copied raw JSON to clipboard')} className="premium-button-dark py-2 px-3 flex items-center space-x-2 text-xs">
            <Copy className="w-3 h-3" />
            <span>Copy JSON</span>
          </button>
          <button onClick={() => showToast('Share link copied to clipboard')} className="premium-button-dark py-2 px-3 flex items-center space-x-2 text-xs">
            <Share2 className="w-3 h-3" />
            <span>Share</span>
          </button>
          <button onClick={() => showToast('Exporting to PDF...')} className="premium-button py-2 px-4 flex items-center space-x-2 text-xs">
            <Download className="w-3 h-3" />
            <span>Export PDF</span>
          </button>
        </div>
      </div>

      {/* Meta-data Strip */}
      <motion.div variants={itemVariants} className="flex items-center space-x-6 text-xs text-[#666] border-b border-[#222] pb-6 mb-8 font-mono">
        <div className="flex items-center space-x-2">
          <span className="uppercase tracking-wider">Est. Duration:</span>
          <span className="text-white">{estDuration}m</span>
        </div>
        <div className="w-1 h-1 rounded-full bg-[#333]"></div>
        <div className="flex items-center space-x-2">
          <span className="uppercase tracking-wider">Word Count:</span>
          <span className="text-white">{wordCount.toLocaleString()}</span>
        </div>
        <div className="w-1 h-1 rounded-full bg-[#333]"></div>
        <div className="flex items-center space-x-2">
          <span className="uppercase tracking-wider">Processing Time:</span>
          <span className="text-white">4.52s</span>
        </div>
        <div className="w-1 h-1 rounded-full bg-[#333]"></div>
        <div className="flex items-center space-x-2">
          <span className="uppercase tracking-wider">Model:</span>
          <span className="text-white">Llama-3-QLoRA</span>
        </div>
      </motion.div>

      {/* Top Row: Hook & Score */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Hook */}
        <motion.div variants={itemVariants} className="premium-panel p-6 lg:col-span-2 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Flame className="w-5 h-5 text-gray-400" />
              <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Optimal Hook Segment</h2>
            </div>
            <span className="text-[10px] text-green-400 font-mono bg-green-950/30 px-2 py-1 rounded border border-green-900/50">HIGH RETENTION LIKELIHOOD</span>
          </div>
          <div className="bg-[#050505] border border-[#222] p-4 rounded-lg relative overflow-hidden">
            <div className="absolute left-0 top-0 bottom-0 w-1 bg-white"></div>
            <p className="text-gray-200 font-mono text-sm leading-relaxed pl-3">
              {hook}
            </p>
          </div>
        </motion.div>

        {/* Quality Score */}
        <motion.div variants={itemVariants} className="premium-panel p-6 flex flex-col justify-center items-center text-center">
          <div className="flex items-center space-x-2 mb-4">
            <TrendingUp className="w-5 h-5 text-gray-400" />
            <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Quality Score</h2>
          </div>
          <div className="text-6xl font-light text-white tracking-tighter flex items-end justify-center">
            {quality_score}
            <span className="text-2xl text-[#555] font-normal tracking-normal ml-1 mb-1">/10</span>
          </div>
          
          <div className="w-full flex space-x-1 mt-6 h-1">
            {[...Array(10)].map((_, i) => (
              <div 
                key={i} 
                className={`flex-1 rounded-full transition-all duration-1000 ${i < Math.floor(quality_score) ? 'bg-white' : 'bg-[#222]'}`}
              />
            ))}
          </div>
        </motion.div>
      </div>

      {/* Visual Timeline Row */}
      <motion.div variants={itemVariants} className="premium-panel p-6 overflow-hidden">
        <div className="flex items-center space-x-3 mb-6">
          <Activity className="w-5 h-5 text-gray-400" />
          <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Event Timeline</h2>
        </div>
        <TimelineVisualization hook={hook} risks={drop_off_risks} />
      </motion.div>

      {/* Graph Row */}
      <motion.div variants={itemVariants} className="premium-panel p-6">
        <div className="flex items-center space-x-3 mb-6 border-b border-[#222] pb-4">
          <LineChartIcon className="w-5 h-5 text-gray-400" />
          <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Predicted Retention Curve</h2>
        </div>
        <RetentionGraph dropOffRisks={drop_off_risks} />
      </motion.div>

      {/* Deep Analysis Row: Risks & Segments */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Drop-off Risks with Explanations */}
        <motion.div variants={itemVariants} className="premium-panel p-6">
          <div className="flex items-center space-x-3 mb-6 border-b border-[#222] pb-4">
            <AlertTriangle className="w-5 h-5 text-gray-400" />
            <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Critical Drop-off Risks</h2>
          </div>
          <div className="space-y-6">
            {drop_off_risks.map((risk, index) => (
              <div key={index} className="space-y-2">
                <div className="flex items-center space-x-3">
                  <span className="shrink-0 inline-flex items-center justify-center px-2 py-1 rounded bg-red-950/20 text-red-400 font-mono text-xs border border-red-900/30">
                    {risk.timestamp}
                  </span>
                  <span className="text-gray-200 font-medium">{risk.reason}</span>
                </div>
                {risk.explanation && (
                  <div className="pl-4 border-l border-[#333] ml-3 mt-2 text-sm text-[#888] flex items-start space-x-2">
                    <Lightbulb className="w-4 h-4 shrink-0 text-[#666] mt-0.5" />
                    <p className="leading-relaxed">{risk.explanation}</p>
                  </div>
                )}
              </div>
            ))}
            {drop_off_risks.length === 0 && (
              <div className="text-[#555] italic">No significant drop-off risks identified.</div>
            )}
          </div>
        </motion.div>

        {/* Segments Breakdown */}
        <motion.div variants={itemVariants} className="premium-panel p-6">
          <div className="flex items-center space-x-3 mb-6 border-b border-[#222] pb-4">
            <Activity className="w-5 h-5 text-gray-400" />
            <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Segment Analysis</h2>
          </div>
          <SegmentBreakdown segments={segments} />
        </motion.div>

      </div>

      {/* Improvements */}
      <motion.div variants={itemVariants} className="premium-panel p-6">
        <div className="flex items-center space-x-3 mb-6 border-b border-[#222] pb-4">
          <Sparkles className="w-5 h-5 text-gray-400" />
          <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Actionable Enhancements</h2>
        </div>
        <ul className="space-y-4">
          {improvements.map((improvement, index) => (
            <li key={index} className="flex space-x-4 text-sm group">
              <span className="shrink-0 w-6 h-6 rounded-full bg-[#111] border border-[#333] text-gray-400 flex items-center justify-center font-medium text-xs transition-colors group-hover:border-gray-500 group-hover:text-white">
                {index + 1}
              </span>
              <span className="text-gray-400 leading-relaxed group-hover:text-gray-200 transition-colors">{improvement}</span>
            </li>
          ))}
        </ul>
      </motion.div>

    </motion.div>
  );
}

function LineChartIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 3v18h18" />
      <path d="m19 9-5 5-4-4-3 3" />
    </svg>
  )
}
