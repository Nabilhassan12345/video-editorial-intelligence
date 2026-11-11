const { segmentTranscript } = require('../utils/segmentTranscript');
const { analyzeTranscript } = require('./openaiService');
const { formatAndValidateOutput } = require('../utils/formatOutput');
const { runPythonPipeline } = require('./pythonRunner');

/**
 * Maps the Python pipeline JSON (PipelineResult.to_dict() from src/pipeline/video_pipeline.py)
 * to the UI schema: { hook, drop_off_risks[], quality_score, improvements[], segments[] }.
 *
 * Python shape: { hook, hook_segment_index, hook_confidence, drop_off_risks[], quality_score,
 *   improvements[], segments[{ start_timestamp, end_timestamp, text, hook_score, risk_score,
 *   quality_score, wpm, silence_ratio }], retention_curve[], metadata{ model_status, ... } }
 */
function formatPythonOutput(pyData) {
  const metadata = pyData.metadata || {};
  const pySegments = Array.isArray(pyData.segments) ? pyData.segments : [];
  const pyRisks = Array.isArray(pyData.drop_off_risks) ? pyData.drop_off_risks : [];

  const drop_off_risks = pyRisks.map(risk => ({
    timestamp: risk.timestamp,
    reason: risk.reason,
    explanation: risk.explanation,
    risk_score: risk.risk_score,
    evidence: risk.evidence || {},
  }));

  const segments = pySegments.map(seg => ({
    start_time: formatSeconds(seg.start_time),
    end_time: formatSeconds(seg.end_time),
    text_summary: seg.text || '',
    risk_score: Math.round((seg.risk_score || 0) * 10),
    engagement_score: Math.round(seg.quality_score || 0),
  }));

  return {
    hook: pyData.hook || 'No specific hook identified.',
    drop_off_risks,
    quality_score: typeof pyData.quality_score === 'number' ? Math.round(pyData.quality_score * 10) / 10 : 0,
    improvements: Array.isArray(pyData.improvements) ? pyData.improvements : [],
    segments,
    source: 'python_model',
    model_status: metadata.model_status || 'unknown',
    prediction_note: metadata.prediction_note || '',
  };
}

/**
 * Formats a number of seconds as "MM:SS".
 */
function formatSeconds(secs) {
  const s = typeof secs === 'number' ? secs : 0;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

/**
 * Main analysis pipeline.
 *
 * Attempts the real Python ML pipeline first; if unavailable (Python not found,
 * model weights missing, timeout, etc.) silently falls back to the OpenAI GPT-4o flow.
 */
async function run(rawTranscript) {
  // --- Attempt 1: Python ML pipeline (PyTorch + PEFT LoRA) ---
  try {
    console.log('[Pipeline] Attempting real Python ML pipeline...');
    const pyData = await runPythonPipeline(rawTranscript);
    const result = formatPythonOutput(pyData);
    console.log('[Pipeline] Python ML pipeline succeeded — returning enriched result.');
    return result;
  } catch (pyErr) {
    console.warn(
      '[Pipeline] Python pipeline unavailable, falling back to OpenAI GPT-4o flow.',
      pyErr.message
    );
  }

  // --- Fallback: OpenAI GPT-4o (original behavior) ---
  // Step 1 & 2: Input & Preprocessing
  console.log('[Pipeline] Step 1 & 2: Segmenting and extracting heuristics...');
  const preprocessedData = segmentTranscript(rawTranscript);

  // Step 3: Analysis (LLM / Mock)
  console.log(`[Pipeline] Step 3: Sending data to LLM (segments: ${preprocessedData.segments.length})...`);
  const rawLlmResponse = await analyzeTranscript(rawTranscript, preprocessedData);

  // Step 4: Output Structuring
  console.log('[Pipeline] Step 4: Formatting and validating output...');
  const finalResult = formatAndValidateOutput(rawLlmResponse);
  finalResult.source = process.env.OPENAI_API_KEY ? 'openai' : 'mock';
  if (finalResult.source === 'mock') {
    finalResult.prediction_note = 'Illustrative mock response: no Python model and no OPENAI_API_KEY were available.';
  }

  return finalResult;
}

module.exports = { run, formatPythonOutput };
