import { Outlet, Link, useLocation } from 'react-router-dom';
import { Activity, LayoutDashboard, History, Cpu, FileVideo, Search, Bell, Settings, Key, CreditCard, ChevronRight } from 'lucide-react';
import { motion } from 'framer-motion';

export default function DashboardLayout() {
  const location = useLocation();

  const mainNavItems = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Analyze', path: '/analyze', icon: FileVideo },
    { name: 'History', path: '/history', icon: History },
    { name: 'Model Insights', path: '/model', icon: Cpu },
  ];

  const workspaceNavItems = [
    { name: 'API Keys', path: '/api-keys', icon: Key, disabled: false },
    { name: 'Billing', path: '#', icon: CreditCard, disabled: true },
    { name: 'Settings', path: '#', icon: Settings, disabled: true },
  ];

  const breadcrumbName = [...mainNavItems, ...workspaceNavItems].find(item => location.pathname.startsWith(item.path))?.name || 'Overview';

  return (
    <div className="min-h-screen bg-[#000000] text-white flex">
      {/* Sidebar */}
      <aside className="w-64 border-r border-[#222] bg-[#050505] hidden md:flex flex-col select-none">
        <div className="p-6 flex items-center space-x-3 cursor-pointer hover:opacity-80 transition-opacity" onClick={() => window.location.href = '/'}>
          <Activity className="w-6 h-6 text-white" />
          <span className="font-bold tracking-tight">Editorial AI</span>
        </div>
        
        <div className="flex-1 overflow-y-auto py-2">
          <div className="px-4 mb-2 text-[10px] font-semibold text-[#555] uppercase tracking-wider">Platform</div>
          <nav className="px-2 space-y-1">
            {mainNavItems.map((item) => {
              const isActive = location.pathname.startsWith(item.path);
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.path}
                  className={`flex items-center space-x-3 px-3 py-2 rounded-lg transition-all duration-200 text-sm ${
                    isActive 
                      ? 'bg-white text-black font-medium' 
                      : 'text-[#888] hover:bg-[#111] hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.name}</span>
                  {isActive && (
                    <motion.div layoutId="activeNav" className="absolute left-0 w-1 h-8 bg-white rounded-r-md hidden" />
                  )}
                </Link>
              );
            })}
          </nav>

          <div className="px-4 mt-8 mb-2 text-[10px] font-semibold text-[#555] uppercase tracking-wider">Workspace</div>
          <nav className="px-2 space-y-1">
            {workspaceNavItems.map((item) => {
              const isActive = location.pathname.startsWith(item.path) && !item.disabled;
              const Icon = item.icon;
              
              if (item.disabled) {
                return (
                  <a
                    key={item.name}
                    href={item.path}
                    className="flex items-center space-x-3 px-3 py-2 rounded-lg text-sm text-[#555] hover:bg-[#111] hover:text-[#888] transition-all cursor-not-allowed"
                    title="Available on Enterprise Plan"
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.name} (Pro)</span>
                  </a>
                );
              }

              return (
                <Link
                  key={item.name}
                  to={item.path}
                  className={`flex items-center space-x-3 px-3 py-2 rounded-lg transition-all duration-200 text-sm ${
                    isActive 
                      ? 'bg-white text-black font-medium' 
                      : 'text-[#888] hover:bg-[#111] hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="p-4 border-t border-[#222] bg-[#050505]">
          <div className="flex items-center justify-between hover:bg-[#111] p-2 -mx-2 rounded-lg transition-colors cursor-pointer group">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-gray-700 to-gray-900 flex items-center justify-center text-xs font-bold border border-[#333]">
                N
              </div>
              <div className="text-sm">
                <p className="text-gray-200 font-medium leading-none mb-1">Nabil Hassan</p>
                <p className="text-[#666] text-xs leading-none">nabil@example.com</p>
              </div>
            </div>
            <Settings className="w-4 h-4 text-[#555] opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-h-screen overflow-hidden">
        {/* Top Header */}
        <header className="h-14 border-b border-[#222] bg-[#000] flex items-center justify-between px-6 shrink-0 relative z-20">
          <div className="flex items-center space-x-2 text-sm text-[#888]">
            <span className="hover:text-white cursor-pointer transition-colors">Workspace</span>
            <ChevronRight className="w-3 h-3" />
            <span className="text-white">{breadcrumbName}</span>
          </div>

          <div className="flex items-center space-x-4">
            <div className="relative hidden md:block">
              <Search className="w-4 h-4 text-[#555] absolute left-3 top-1/2 -translate-y-1/2" />
              <input 
                type="text" 
                placeholder="Search resources... (Cmd+K)" 
                className="bg-[#050505] border border-[#222] text-sm text-white rounded-md pl-9 pr-4 py-1.5 focus:outline-none focus:border-[#444] w-64 transition-colors"
              />
            </div>
            <button className="text-[#555] hover:text-white transition-colors relative">
              <Bell className="w-5 h-5" />
              <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full border border-black"></span>
            </button>
          </div>
        </header>

        {/* Mobile Header (Hidden on Desktop) */}
        <header className="md:hidden border-b border-[#222] p-4 flex items-center justify-between bg-[#050505] shrink-0">
          <div className="flex items-center space-x-3">
            <Activity className="w-6 h-6 text-white" />
            <span className="font-bold tracking-tight">Editorial AI</span>
          </div>
          <Bell className="w-5 h-5 text-[#888]" />
        </header>
        
        <div className="flex-1 overflow-y-auto p-6 md:p-10 relative">
          {/* Subtle background glow for main area */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-3xl h-64 bg-white/[0.015] blur-[100px] pointer-events-none"></div>
          
          <div className="max-w-6xl mx-auto relative z-10">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
}
