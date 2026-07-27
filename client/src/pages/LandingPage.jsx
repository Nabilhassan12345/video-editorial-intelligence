import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Activity, ArrowRight, Zap, Target, LineChart } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden flex flex-col">
      {/* Navbar */}
      <nav className="flex items-center justify-between p-6 md:px-12 relative z-20">
        <div className="flex items-center space-x-3">
          <Activity className="w-6 h-6 text-white" />
          <span className="font-bold tracking-tight text-lg">Editorial AI</span>
        </div>
        <div className="space-x-6">
          <Link to="/analyze" className="text-sm font-medium text-gray-300 hover:text-white transition-colors">Product</Link>
          <Link to="/model" className="text-sm font-medium text-gray-300 hover:text-white transition-colors">Models</Link>
          <Link to="/dashboard" className="text-sm font-medium bg-white text-black px-4 py-2 rounded-full hover:bg-gray-200 transition-colors">
            Go to App
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <main className="flex-1 flex flex-col justify-center items-center text-center px-4 relative z-10 -mt-20">
        {/* Glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-white/[0.03] blur-[120px] rounded-full pointer-events-none"></div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-4xl space-y-8"
        >
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full border border-[#333] bg-[#111] text-xs font-mono text-gray-400 mb-4">
            <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
            <span>v2.0 Model Now Available</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold tracking-tighter leading-tight">
            Predict retention. <br className="hidden md:block" />
            <span className="text-gray-400">Optimize content.</span> Grow faster.
          </h1>
          
          <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto font-light leading-relaxed">
            The first machine learning pipeline designed specifically for video editors and creators to scientifically maximize viewer engagement.
          </p>

          <div className="pt-4">
            <Link to="/dashboard" className="inline-flex items-center justify-center space-x-2 bg-white text-black px-8 py-4 rounded-lg font-medium text-lg hover:bg-gray-200 transition-colors shadow-[0_0_40px_rgba(255,255,255,0.2)]">
              <span>Start Analyzing</span>
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl w-full mt-32 text-left">
          <FeatureCard 
            icon={<Target className="w-6 h-6" />}
            title="Hook Detection"
            description="Algorithmic identification of the most powerful retention hooks in your raw transcript."
          />
          <FeatureCard 
            icon={<LineChart className="w-6 h-6" />}
            title="Drop-off Prediction"
            description="Anticipate viewer churn before publishing with our simulated retention decay models."
          />
          <FeatureCard 
            icon={<Zap className="w-6 h-6" />}
            title="AI Explanations"
            description="Get deep semantic analysis on why specific segments fail to hold audience attention."
          />
        </div>
      </main>
      
      <footer className="border-t border-[#222] py-8 text-center text-sm text-gray-500 relative z-20">
        <p>© {new Date().getFullYear()} Editorial AI Inc. All rights reserved.</p>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <div className="premium-panel p-6 bg-[#050505]">
      <div className="w-12 h-12 rounded-lg bg-[#111] border border-[#333] flex items-center justify-center mb-6 text-white">
        {icon}
      </div>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-gray-400 text-sm leading-relaxed">{description}</p>
    </div>
  );
}
