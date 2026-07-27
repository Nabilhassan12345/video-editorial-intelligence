require('dotenv').config();
const express = require('express');
const cors = require('cors');

const analyzeRoute = require('./routes/analyze');
const historyRoute = require('./routes/history');
const resultsRoute = require('./routes/results');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/analyze', analyzeRoute);
app.use('/history', historyRoute);
app.use('/results', resultsRoute);

// Basic health check
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'healthy', service: 'video-editorial-intelligence' });
});

app.listen(PORT, () => {
  console.log(`[Server] Video Editorial Intelligence backend running on port ${PORT}`);
});
