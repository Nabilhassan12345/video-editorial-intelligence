const { segmentTranscript } = require('../utils/segmentTranscript');
const { analyzeTranscript } = require('./openaiService');
const { formatAndValidateOutput } = require('../utils/formatOutput');

/**
 * Simulates a Machine Learning pipeline for transcript analysis.
 */
async function run(rawTranscript) {
  // Step 1 & 2: Input & Preprocessing
  console.log('[Pipeline] Step 1 & 2: Segmenting and extracting heuristics...');
  const preprocessedData = segmentTranscript(rawTranscript);

  // Step 3: Analysis (LLM / Mock)
  console.log(`[Pipeline] Step 3: Sending data to LLM (segments: ${preprocessedData.segments.length})...`);
  const rawLlmResponse = await analyzeTranscript(rawTranscript, preprocessedData);

  // Step 4: Output Structuring
  console.log('[Pipeline] Step 4: Formatting and validating output...');
  const finalResult = formatAndValidateOutput(rawLlmResponse);

  return finalResult;
}

module.exports = { run };
