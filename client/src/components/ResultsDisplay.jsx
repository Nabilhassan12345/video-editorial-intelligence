import { motion } from 'framer-motion';
import { Flame, AlertTriangle, TrendingUp, Sparkles } from 'lucide-react';
import RetentionGraph from './RetentionGraph';

export default function ResultsDisplay({ results }) {
  const { hook, drop_off_risks, quality_score, improvements } = results;

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } }
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
        <motion.div variants={itemVariants} className="glass-panel p-6 lg:col-span-2 relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-32 h-32 bg-orange-500/10 rounded-full blur-3xl -mr-10 -mt-10 transition-all duration-500 group-hover:bg-orange-500/20"></div>
          <div className="flex items-center space-x-3 mb-4 relative z-10">
            <div className="p-2 bg-orange-500/20 rounded-lg">
              <Flame className="w-5 h-5 text-orange-400" />
            </div>
            <h2 className="text-xl font-bold text-white tracking-wide">Strongest Hook</h2>
          </div>
          <p className="text-gray-300 font-mono text-sm leading-relaxed border-l-2 border-orange-500/50 pl-4 py-1 relative z-10">
            {hook}
          </p>
        </motion.div>

        {/* Quality Score */}
        <motion.div variants={itemVariants} className="glass-panel p-6 flex flex-col justify-center items-center text-center relative overflow-hidden group">
          <div className="absolute bottom-0 left-0 w-full h-1/2 bg-indigo-500/10 blur-2xl transition-all duration-500 group-hover:bg-indigo-500/20"></div>
          <div className="flex items-center space-x-2 mb-2 relative z-10">
            <TrendingUp className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-semibold text-gray-200">Quality Score</h2>
          </div>
          <div className="relative z-10 mt-2">
            <div className="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-gray-500">
              {quality_score}
              <span className="text-2xl text-gray-500 font-medium">/10</span>
            </div>
          </div>
          
          {/* Progress Bar */}
          <div className="w-full h-2 bg-gray-800 rounded-full mt-6 relative z-10 overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${(quality_score / 10) * 100}%` }}
              transition={{ duration: 1.5, ease: "easeOut", delay: 0.5 }}
              className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
            />
          </div>
        </motion.div>
      </div>

      {/* Graph Row */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <ActivityIcon className="w-5 h-5 text-blue-400" />
            </div>
            <h2 className="text-xl font-bold text-white tracking-wide">Predicted Retention Curve</h2>
          </div>
        </div>
        <RetentionGraph dropOffRisks={drop_off_risks} />
      </motion.div>

      {/* Bottom Row: Risks & Improvements */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Drop-off Risks */}
        <motion.div variants={itemVariants} className="glass-panel p-6 border-red-500/20 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-red-500/5 rounded-full blur-3xl"></div>
          <div className="flex items-center space-x-3 mb-6 relative z-10">
            <div className="p-2 bg-red-500/20 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-red-400" />
            </div>
            <h2 className="text-xl font-bold text-white tracking-wide">Drop-off Risks</h2>
          </div>
          <ul className="space-y-4 relative z-10">
            {drop_off_risks.map((risk, index) => (
              <li key={index} className="flex space-x-3 text-sm">
                <span className="shrink-0 inline-flex items-center justify-center px-2 py-1 rounded bg-red-500/10 text-red-400 font-mono text-xs border border-red-500/20 font-semibold h-fit mt-0.5">
                  {risk.timestamp}
                </span>
                <span className="text-gray-300 leading-relaxed">{risk.reason}</span>
              </li>
            ))}
            {drop_off_risks.length === 0 && (
              <li className="text-gray-500 italic">No significant drop-off risks identified.</li>
            )}
          </ul>
        </motion.div>

        {/* Improvements */}
        <motion.div variants={itemVariants} className="glass-panel p-6 relative overflow-hidden">
          <div className="absolute bottom-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-3xl"></div>
          <div className="flex items-center space-x-3 mb-6 relative z-10">
            <div className="p-2 bg-emerald-500/20 rounded-lg">
              <Sparkles className="w-5 h-5 text-emerald-400" />
            </div>
            <h2 className="text-xl font-bold text-white tracking-wide">Actionable Improvements</h2>
          </div>
          <ul className="space-y-4 relative z-10">
            {improvements.map((improvement, index) => (
              <li key={index} className="flex space-x-3 text-sm">
                <span className="shrink-0 w-6 h-6 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs">
                  {index + 1}
                </span>
                <span className="text-gray-300 leading-relaxed">{improvement}</span>
              </li>
            ))}
          </ul>
        </motion.div>

      </div>
    </motion.div>
  );
}

// Inline Activity Icon since we need it above
function ActivityIcon(props) {
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
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  )
}
