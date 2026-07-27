// In-memory database for storing analyses
const analyses = new Map();

function saveAnalysis(id, data) {
  const record = {
    id,
    ...data,
    createdAt: new Date().toISOString()
  };
  analyses.set(id, record);
  return record;
}

function getAnalysis(id) {
  return analyses.get(id);
}

function getAllAnalyses() {
  return Array.from(analyses.values()).sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
}

module.exports = {
  saveAnalysis,
  getAnalysis,
  getAllAnalyses
};
