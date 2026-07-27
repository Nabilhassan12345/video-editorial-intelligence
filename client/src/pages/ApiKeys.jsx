import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Key, Plus, Trash2, Copy, CheckCircle2, Eye, EyeOff, Activity, AlertCircle } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// Mock API request data
const mockUsageData = Array.from({ length: 30 }).map((_, i) => ({
  day: `Day ${i + 1}`,
  requests: Math.floor(Math.random() * 5000) + 1000,
}));

export default function ApiKeys() {
  const [keys, setKeys] = useState([
    { id: '1', name: 'Production Backend', prefix: 'sk_live_...9f2a', created: '2025-10-12', lastUsed: '2 mins ago' },
    { id: '2', name: 'Development Local', prefix: 'sk_test_...8b1c', created: '2026-01-05', lastUsed: '5 days ago' }
  ]);
  const [isCreating, setIsCreating] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [newlyGeneratedKey, setNewlyGeneratedKey] = useState(null);
  const [copied, setCopied] = useState(false);

  const handleCreate = (e) => {
    e.preventDefault();
    if (!newKeyName.trim()) return;
    
    // Simulate generation
    const fullKey = `sk_live_${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`;
    const prefix = `sk_live_...${fullKey.slice(-4)}`;
    
    setKeys(prev => [{
      id: Date.now().toString(),
      name: newKeyName,
      prefix,
      created: new Date().toISOString().split('T')[0],
      lastUsed: 'Never'
    }, ...prev]);
    
    setNewlyGeneratedKey(fullKey);
    setNewKeyName('');
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDelete = (id) => {
    setKeys(keys.filter(k => k.id !== id));
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500 pb-20">
      <header className="space-y-2">
        <div className="flex items-center space-x-3 text-[#888] mb-2">
          <Key className="w-5 h-5" />
          <span className="text-sm font-semibold uppercase tracking-wider">Developer Access</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-white">
          API Keys
        </h1>
        <p className="text-[#888] max-w-2xl font-light">
          Manage your API keys for programmatic access to the Editorial AI inference pipeline. Do not share your secret keys in publicly accessible areas.
        </p>
      </header>

      {/* Usage Chart */}
      <div className="premium-panel p-6">
        <div className="flex items-center justify-between mb-6 border-b border-[#222] pb-4">
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-gray-400" />
            <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">API Usage (Last 30 Days)</h2>
          </div>
          <div className="text-right">
            <div className="text-2xl font-light text-white">124,592</div>
            <div className="text-[10px] text-green-400 font-mono">+14% vs last month</div>
          </div>
        </div>
        <div className="h-48 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={mockUsageData}>
              <defs>
                <linearGradient id="usageGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ffffff" stopOpacity={0.2}/>
                  <stop offset="95%" stopColor="#ffffff" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <Tooltip 
                contentStyle={{ backgroundColor: '#111', borderColor: '#333', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
              <Area 
                type="monotone" 
                dataKey="requests" 
                stroke="#fff" 
                strokeWidth={2}
                fillOpacity={1} 
                fill="url(#usageGradient)" 
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Key Management */}
      <div className="premium-panel p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold flex items-center space-x-2">
            <span>Secret Keys</span>
          </h2>
          <button 
            onClick={() => setIsCreating(true)}
            className="premium-button text-xs py-2 px-4 flex items-center space-x-2"
            disabled={isCreating}
          >
            <Plus className="w-4 h-4" />
            <span>Create new secret key</span>
          </button>
        </div>

        <AnimatePresence>
          {isCreating && !newlyGeneratedKey && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-[#050505] border border-[#333] rounded-lg p-4 mb-6 overflow-hidden"
            >
              <form onSubmit={handleCreate} className="flex items-end gap-4">
                <div className="flex-1 space-y-2">
                  <label className="block text-xs font-semibold text-[#888] uppercase tracking-wider">Key Name</label>
                  <input 
                    type="text" 
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                    placeholder="e.g. CI/CD Pipeline" 
                    className="premium-input py-2"
                    autoFocus
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <button type="button" onClick={() => setIsCreating(false)} className="premium-button-dark py-2 px-4 text-xs">Cancel</button>
                  <button type="submit" className="premium-button py-2 px-4 text-xs">Generate</button>
                </div>
              </form>
            </motion.div>
          )}

          {newlyGeneratedKey && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-[#0a0a0a] border border-green-900/50 rounded-lg p-6 mb-6 shadow-[0_0_30px_rgba(34,197,94,0.1)] relative overflow-hidden"
            >
              <div className="absolute top-0 left-0 w-1 h-full bg-green-500"></div>
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-green-400 font-semibold mb-2 flex items-center space-x-2">
                    <CheckCircle2 className="w-5 h-5" />
                    <span>Save your secret key</span>
                  </h3>
                  <p className="text-sm text-gray-400 mb-4 max-w-xl">
                    Please save this secret key somewhere safe and accessible. For security reasons, <strong>you won't be able to view it again</strong> through your Editorial AI account. If you lose this secret key, you'll need to generate a new one.
                  </p>
                  <div className="flex items-center space-x-3 bg-[#000] border border-[#333] p-3 rounded-lg">
                    <code className="text-gray-200 font-mono text-sm flex-1 select-all">{newlyGeneratedKey}</code>
                    <button 
                      onClick={() => copyToClipboard(newlyGeneratedKey)}
                      className="p-2 hover:bg-[#222] rounded transition-colors text-gray-400 hover:text-white"
                    >
                      {copied ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
                <button 
                  onClick={() => {
                    setNewlyGeneratedKey(null);
                    setIsCreating(false);
                  }}
                  className="premium-button-dark py-2 px-4 text-xs shrink-0"
                >
                  Done
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-[#050505] text-[#888] uppercase text-xs border-b border-[#222]">
              <tr>
                <th className="px-6 py-4 font-medium">Name</th>
                <th className="px-6 py-4 font-medium">Secret Key</th>
                <th className="px-6 py-4 font-medium">Created</th>
                <th className="px-6 py-4 font-medium">Last Used</th>
                <th className="px-6 py-4 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#222]">
              {keys.map((key) => (
                <tr key={key.id} className="hover:bg-[#080808] transition-colors group">
                  <td className="px-6 py-4 text-gray-200 font-medium">{key.name}</td>
                  <td className="px-6 py-4 font-mono text-xs text-gray-400">{key.prefix}</td>
                  <td className="px-6 py-4 text-gray-400">{key.created}</td>
                  <td className="px-6 py-4 text-gray-400">{key.lastUsed}</td>
                  <td className="px-6 py-4 text-right">
                    <button 
                      onClick={() => handleDelete(key.id)}
                      className="p-2 text-[#555] hover:text-red-400 hover:bg-red-500/10 rounded transition-colors opacity-0 group-hover:opacity-100"
                      title="Revoke Key"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
              {keys.length === 0 && (
                <tr>
                  <td colSpan="5" className="px-6 py-8 text-center text-[#555] italic">
                    No API keys generated yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
