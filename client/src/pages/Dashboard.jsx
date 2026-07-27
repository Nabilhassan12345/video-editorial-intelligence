import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Plus, BarChart2, Star, Clock } from 'lucide-react';

export default function Dashboard() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

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

  const avgQuality = history.length > 0 
    ? (history.reduce((acc, curr) => acc + curr.quality_score, 0) / history.length).toFixed(1)
    : '-';

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight mb-2">Overview</h1>
          <p className="text-[#888]">Analyze and optimize your video content performance.</p>
        </div>
        <Link to="/analyze" className="premium-button">
          <Plus className="w-4 h-4 mr-2" />
          New Analysis
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard icon={<BarChart2 />} title="Total Analyses" value={history.length} />
        <StatCard icon={<Star />} title="Avg. Quality Score" value={avgQuality} />
        <StatCard icon={<Clock />} title="Avg. Retention" value="~62%" />
      </div>

      <div className="pt-8">
        <h2 className="text-xl font-semibold mb-6">Recent Analyses</h2>
        
        {loading ? (
          <div className="text-[#888] text-sm animate-pulse">Loading data...</div>
        ) : history.length === 0 ? (
          <div className="premium-panel p-8 text-center border-dashed">
            <p className="text-[#888] mb-4">No analyses yet.</p>
            <Link to="/analyze" className="premium-button-dark inline-flex">Run your first analysis</Link>
          </div>
        ) : (
          <div className="premium-panel overflow-hidden">
            <table className="w-full text-sm text-left">
              <thead className="bg-[#050505] text-[#888] uppercase text-xs border-b border-[#222]">
                <tr>
                  <th className="px-6 py-4 font-medium">ID</th>
                  <th className="px-6 py-4 font-medium">Date</th>
                  <th className="px-6 py-4 font-medium">Quality</th>
                  <th className="px-6 py-4 font-medium text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#222]">
                {history.slice(0, 5).map((item) => (
                  <tr key={item.id} className="hover:bg-[#080808] transition-colors">
                    <td className="px-6 py-4 font-mono text-xs text-gray-300">{item.id.slice(0,8)}...</td>
                    <td className="px-6 py-4 text-gray-400">{new Date(item.createdAt).toLocaleDateString()}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        <div className="w-12 bg-[#222] h-1.5 rounded-full overflow-hidden">
                          <div className="bg-white h-full" style={{ width: `${(item.quality_score/10)*100}%` }}></div>
                        </div>
                        <span>{item.quality_score}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <Link to={`/results/${item.id}`} className="text-white hover:underline text-xs">View Report &rarr;</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ icon, title, value }) {
  return (
    <div className="premium-panel p-6 bg-[#050505]">
      <div className="flex items-center space-x-3 mb-4 text-[#888]">
        {icon}
        <span className="text-sm font-medium">{title}</span>
      </div>
      <div className="text-4xl font-light tracking-tight">{value}</div>
    </div>
  );
}
