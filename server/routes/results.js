const express = require('express');
const router = express.Router();
const { getAnalysis } = require('../models/db');

router.get('/:id', (req, res) => {
  try {
    const { id } = req.params;
    const analysis = getAnalysis(id);
    
    if (!analysis) {
      return res.status(404).json({ error: 'Analysis not found.' });
    }
    
    res.json(analysis);
  } catch (error) {
    console.error('[Results Route Error]', error);
    res.status(500).json({ error: 'Failed to fetch result.' });
  }
});

module.exports = router;
