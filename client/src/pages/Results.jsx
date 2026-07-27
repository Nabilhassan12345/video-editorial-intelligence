import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Flame, AlertTriangle, TrendingUp, Sparkles, Activity, ArrowLeft, Lightbulb } from 'lucide-react';
import RetentionGraph from '../components/RetentionGraph';
import TimelineVisualization from '../components/TimelineVisualization';
import SegmentBreakdown from '../components/SegmentBreakdown';

export default function Results() {
  const { id } = useParams();
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

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

  const { hook, drop_off_risks, quality_score, improvements, segments } = results;

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
      className="space-y-6 pb-20"
    >
      <div className="flex items-center space-x-4 mb-8">
        <Link to="/history" className="w-8 h-8 rounded-full bg-[#111] border border-[#333] flex items-center justify-center hover:bg-[#222] transition-colors text-white">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Analysis Report</h1>
          <p className="text-[#666] text-xs font-mono mt-1">ID: {id}</p>
        </div>
      </div>

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
          <div className="text-6xl font-light text-white tracking-tighter">
            {quality_score}
            <span className="text-2xl text-[#555] font-normal tracking-normal ml-1">/10</span>
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
