import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { History as HistoryIcon, Search } from 'lucide-react';

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetch('http://localhost:3001/history')
      .then(res => res.json())
      .then(data => {
        setHistory(data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const filteredHistory = history.filter(item => 
    item.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.hook.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center space-x-3 text-[#888] mb-2">
            <HistoryIcon className="w-5 h-5" />
            <span className="text-sm font-semibold uppercase tracking-wider">Analysis Log</span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white">
            Past Analyses
          </h1>
        </div>
        
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[#666]" />
          <input 
            type="text" 
            placeholder="Search by ID or Hook..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-9 pr-4 py-2 bg-[#050505] border border-[#222] rounded-lg text-sm text-gray-200 focus:outline-none focus:border-[#444] w-full md:w-64 transition-colors"
          />
        </div>
      </div>

      <div className="premium-panel overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-[#888] animate-pulse">Loading history...</div>
        ) : filteredHistory.length === 0 ? (
          <div className="p-8 text-center text-[#888]">No matching analyses found.</div>
        ) : (
          <table className="w-full text-sm text-left">
            <thead className="bg-[#050505] text-[#888] uppercase text-xs border-b border-[#222]">
              <tr>
                <th className="px-6 py-4 font-medium">ID</th>
                <th className="px-6 py-4 font-medium">Date</th>
                <th className="px-6 py-4 font-medium max-w-[300px]">Hook Preview</th>
                <th className="px-6 py-4 font-medium text-center">Score</th>
                <th className="px-6 py-4 font-medium text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#222]">
              {filteredHistory.map((item) => (
                <tr key={item.id} className="hover:bg-[#080808] transition-colors group">
                  <td className="px-6 py-4 font-mono text-xs text-gray-400">{item.id.slice(0,8)}...</td>
                  <td className="px-6 py-4 text-gray-400">{new Date(item.createdAt).toLocaleString()}</td>
                  <td className="px-6 py-4 text-gray-300 truncate max-w-[300px]">
                    {item.hook}
                  </td>
                  <td className="px-6 py-4 text-center">
                    <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full border ${item.quality_score >= 7 ? 'border-green-900/50 bg-green-950/20 text-green-400' : 'border-[#333] bg-[#111] text-gray-300'}`}>
                      {item.quality_score}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <Link to={`/results/${item.id}`} className="premium-button text-xs py-1.5 px-3 opacity-0 group-hover:opacity-100 transition-opacity inline-flex">
                      View
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
