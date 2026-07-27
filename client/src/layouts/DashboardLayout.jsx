import { Outlet, Link, useLocation } from 'react-router-dom';
import { Activity, LayoutDashboard, History, Cpu, FileVideo } from 'lucide-react';
import { motion } from 'framer-motion';

export default function DashboardLayout() {
  const location = useLocation();

  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Analyze', path: '/analyze', icon: FileVideo },
    { name: 'History', path: '/history', icon: History },
    { name: 'Model Insights', path: '/model', icon: Cpu },
  ];

  return (
    <div className="min-h-screen bg-[#000000] text-white flex">
      {/* Sidebar */}
      <aside className="w-64 border-r border-[#222] bg-[#050505] hidden md:flex flex-col">
        <div className="p-6 flex items-center space-x-3">
          <Activity className="w-6 h-6 text-white" />
          <span className="font-bold tracking-tight">Editorial AI</span>
        </div>
        
        <nav className="flex-1 px-4 space-y-2 mt-4">
          {navItems.map((item) => {
            const isActive = location.pathname.startsWith(item.path);
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.path}
                className={`flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-all duration-200 ${
                  isActive 
                    ? 'bg-white text-black font-medium' 
                    : 'text-[#888] hover:bg-[#111] hover:text-white'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{item.name}</span>
                {isActive && (
                  <motion.div layoutId="activeNav" className="absolute left-0 w-1 h-8 bg-white rounded-r-md hidden" />
                )}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-[#222]">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-full bg-[#222] flex items-center justify-center text-xs font-bold">
              U
            </div>
            <div className="text-sm">
              <p className="text-gray-200">User Workspace</p>
              <p className="text-[#666] text-xs">Pro Plan</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-h-screen overflow-hidden">
        {/* Mobile Header (Hidden on Desktop) */}
        <header className="md:hidden border-b border-[#222] p-4 flex items-center space-x-3 bg-[#050505]">
          <Activity className="w-6 h-6 text-white" />
          <span className="font-bold tracking-tight">Editorial AI</span>
        </header>
        
        <div className="flex-1 overflow-y-auto p-6 md:p-10 relative">
          {/* Subtle background glow for main area */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-3xl h-64 bg-white/[0.02] blur-[100px] pointer-events-none"></div>
          
          <div className="max-w-6xl mx-auto relative z-10">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
}
