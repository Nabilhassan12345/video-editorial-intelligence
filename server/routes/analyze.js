const express = require('express');
const router = express.Router();
const crypto = require('crypto');
const analysisPipeline = require('../services/analysisPipeline');
const { saveAnalysis } = require('../models/db');

router.post('/', async (req, res) => {
  try {
    const { transcript } = req.body;

    if (!transcript || typeof transcript !== 'string' || transcript.trim() === '') {
      return res.status(400).json({ error: 'A valid transcript string is required.' });
    }

    console.log(`[Pipeline] Started analysis for transcript (length: ${transcript.length})`);
    
    // Run the pipeline
    const result = await analysisPipeline.run(transcript);

    // Save to in-memory DB with a unique ID
    const id = crypto.randomUUID();
    const savedRecord = saveAnalysis(id, { transcript, ...result });

    res.json(savedRecord);
  } catch (error) {
    console.error('[Pipeline Error]', error);
    res.status(500).json({ error: 'An error occurred during transcript analysis.' });
  }
});

module.exports = router;
