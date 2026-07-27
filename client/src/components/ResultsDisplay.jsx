import { motion } from 'framer-motion';
import { Flame, AlertTriangle, TrendingUp, Sparkles, Activity } from 'lucide-react';
import RetentionGraph from './RetentionGraph';

export default function ResultsDisplay({ results }) {
  const { hook, drop_off_risks, quality_score, improvements } = results;

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.05 }
    }
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
      className="space-y-6"
    >
      {/* Top Row: Hook & Score */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Hook */}
        <motion.div variants={itemVariants} className="premium-panel p-6 lg:col-span-2 flex flex-col justify-between">
          <div className="flex items-center space-x-3 mb-6">
            <Flame className="w-5 h-5 text-gray-400" />
            <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Optimal Hook Segment</h2>
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
          
          {/* Segmented Progress Bar */}
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

      {/* Graph Row */}
      <motion.div variants={itemVariants} className="premium-panel p-6">
        <div className="flex items-center space-x-3 mb-6 border-b border-[#222] pb-4">
          <Activity className="w-5 h-5 text-gray-400" />
          <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Predicted Retention Curve</h2>
        </div>
        <RetentionGraph dropOffRisks={drop_off_risks} />
      </motion.div>

      {/* Bottom Row: Risks & Improvements */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Drop-off Risks */}
        <motion.div variants={itemVariants} className="premium-panel p-6">
          <div className="flex items-center space-x-3 mb-6">
            <AlertTriangle className="w-5 h-5 text-gray-400" />
            <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Critical Drop-off Risks</h2>
          </div>
          <ul className="space-y-4">
            {drop_off_risks.map((risk, index) => (
              <li key={index} className="flex space-x-4 text-sm group">
                <span className="shrink-0 inline-flex items-center justify-center px-2 py-1 rounded bg-[#111] text-gray-300 font-mono text-xs border border-[#333] transition-colors group-hover:border-gray-500 h-fit mt-0.5">
                  {risk.timestamp}
                </span>
                <span className="text-gray-400 leading-relaxed group-hover:text-gray-200 transition-colors">{risk.reason}</span>
              </li>
            ))}
            {drop_off_risks.length === 0 && (
              <li className="text-[#555] italic">No significant drop-off risks identified.</li>
            )}
          </ul>
        </motion.div>

        {/* Improvements */}
        <motion.div variants={itemVariants} className="premium-panel p-6">
          <div className="flex items-center space-x-3 mb-6">
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

      </div>
    </motion.div>
  );
}
