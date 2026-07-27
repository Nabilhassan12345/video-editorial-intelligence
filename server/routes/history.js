const express = require('express');
const router = express.Router();
const { getAllAnalyses } = require('../models/db');

router.get('/', (req, res) => {
  try {
    const history = getAllAnalyses();
    res.json(history);
  } catch (error) {
    console.error('[History Route Error]', error);
    res.status(500).json({ error: 'Failed to fetch history.' });
  }
});

module.exports = router;
