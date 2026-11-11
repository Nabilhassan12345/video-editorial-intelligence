import { Cpu, Database, Network, GitCommit } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ModelInsights() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8 animate-in fade-in duration-500 pb-20"
    >
      <header className="space-y-2">
        <div className="flex items-center space-x-3 text-[#888] mb-2">
          <Cpu className="w-5 h-5" />
          <span className="text-sm font-semibold uppercase tracking-wider">System Architecture</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-white">
          Model Insights
        </h1>
        <p className="text-[#888] max-w-2xl font-light">
          The local machine learning pipeline, backbone architecture, and its (synthetic-data) evaluation results.
        </p>
      </header>

      {/* Pipeline Diagram */}
      <motion.div variants={itemVariants} className="premium-panel p-8 overflow-hidden relative">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-white to-transparent opacity-20"></div>
        <h2 className="text-lg font-semibold mb-8 flex items-center space-x-2">
          <Network className="w-5 h-5 text-gray-400" />
          <span>Inference Pipeline</span>
        </h2>
        
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 md:gap-0 relative z-10">
          
          <PipelineNode title="Data Ingestion" desc="Transcript & Audio Sync" delay="0s" />
          <div className="hidden md:block h-px bg-[#333] flex-1 mx-4 relative"><div className="absolute top-1/2 right-0 w-2 h-2 rounded-full bg-[#555] -translate-y-1/2"></div></div>
          
          <PipelineNode title="Feature Eng." desc="Lexical Density & Pacing" delay="0.2s" />
          <div className="hidden md:block h-px bg-[#333] flex-1 mx-4 relative"><div className="absolute top-1/2 right-0 w-2 h-2 rounded-full bg-[#555] -translate-y-1/2"></div></div>
          
          <PipelineNode title="Transformer" desc="Contextual Embeddings" delay="0.4s" />
          <div className="hidden md:block h-px bg-[#333] flex-1 mx-4 relative"><div className="absolute top-1/2 right-0 w-2 h-2 rounded-full bg-white -translate-y-1/2 shadow-[0_0_10px_#fff]"></div></div>
          
          <PipelineNode title="Output Head" desc="Retention & Risk Scoring" delay="0.6s" highlight />

        </div>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Details */}
        <motion.div variants={itemVariants} className="premium-panel p-6">
          <h2 className="text-lg font-semibold mb-6 flex items-center space-x-2 border-b border-[#222] pb-4">
            <Database className="w-5 h-5 text-gray-400" />
            <span>Architecture Specifications</span>
          </h2>
          <ul className="space-y-4">
            <SpecItem label="Base Model" value="distilbert-base-uncased" />
            <SpecItem label="Fine-tuning Method" value="PEFT LoRA (q_lin, v_lin)" />
            <SpecItem label="Prediction Heads" value="Hook / Drop-off / Quality" />
            <SpecItem label="Training Dataset" value="18,150 real samples: TTCC + vRetention + YouTube captions + TikTok hooks" />
            <SpecItem label="Inference" value="PyTorch on CPU" />
          </ul>
        </motion.div>

        {/* Metrics */}
        <motion.div variants={itemVariants} className="premium-panel p-6">
          <h2 className="text-lg font-semibold mb-6 flex items-center space-x-2 border-b border-[#222] pb-4">
            <GitCommit className="w-5 h-5 text-gray-400" />
            <span>Held-out Test (2,787 samples, group-aware split) — per-source</span>
          </h2>
          <div className="space-y-6">
            <MetricBar label="Drop-off ROC-AUC (TTCC / vRetention)" value="0.64 / 0.62" isText />
            <MetricBar label="Drop-off PR-AUC (TTCC / vRetention)" value="0.81 / 0.87" isText />
            <MetricBar label="Hook ROC-AUC (TTCC, F1 0.38)" value="0.66" isText />
            <MetricBar label="Quality RMSE (TTCC, Pearson r 0.08)" value="0.99" isText />
            <MetricBar label="CPU Latency" value="~65 ms / segment" isText />
          </div>
          <p className="text-[11px] text-[#666] mt-6 leading-relaxed">
            Real measured retention curves (TTCC, vRetention) + engagement-labeled TikTok hooks; see
            docs/DATASETS.md + docs/EVALUATION.md. Modest, real signal — a baseline to beat, not production accuracy.
          </p>
        </motion.div>
      </div>

    </motion.div>
  );
}

function PipelineNode({ title, desc, highlight }) {
  return (
    <div className={`w-48 p-4 rounded-lg border text-center relative ${highlight ? 'bg-white text-black border-white shadow-[0_0_20px_rgba(255,255,255,0.1)]' : 'bg-[#111] border-[#333] text-gray-300'}`}>
      <div className={`font-semibold text-sm mb-1 ${highlight ? 'text-black' : 'text-white'}`}>{title}</div>
      <div className={`text-[10px] uppercase tracking-wider ${highlight ? 'text-gray-700' : 'text-gray-500'}`}>{desc}</div>
    </div>
  );
}

function SpecItem({ label, value }) {
  return (
    <li className="flex justify-between items-center text-sm">
      <span className="text-gray-400">{label}</span>
      <span className="text-white font-mono bg-[#111] px-2 py-1 rounded border border-[#333]">{value}</span>
    </li>
  );
}

function MetricBar({ label, value, isText }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-2">
        <span className="text-gray-300">{label}</span>
        <span className="text-white font-mono">{isText ? value : `${value}%`}</span>
      </div>
      {!isText && (
        <div className="h-1.5 w-full bg-[#222] rounded-full overflow-hidden">
          <div className="h-full bg-white rounded-full" style={{ width: `${value}%` }}></div>
        </div>
      )}
    </div>
  );
}
