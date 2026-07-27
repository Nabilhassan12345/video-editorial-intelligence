import { BrowserRouter, Routes, Route } from 'react-router-dom';
import DashboardLayout from './layouts/DashboardLayout';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import Analyze from './pages/Analyze';
import Results from './pages/Results';
import History from './pages/History';
import ModelInsights from './pages/ModelInsights';
import ApiKeys from './pages/ApiKeys';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        
        <Route element={<DashboardLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/results/:id" element={<Results />} />
          <Route path="/history" element={<History />} />
          <Route path="/model" element={<ModelInsights />} />
          <Route path="/api-keys" element={<ApiKeys />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
