function formatAndValidateOutput(llmResponse) {
  return {
    hook: llmResponse.hook || "No specific hook identified.",
    drop_off_risks: Array.isArray(llmResponse.drop_off_risks) ? llmResponse.drop_off_risks : [],
    quality_score: typeof llmResponse.quality_score === 'number' ? llmResponse.quality_score : 5,
    improvements: Array.isArray(llmResponse.improvements) ? llmResponse.improvements : [],
    segments: Array.isArray(llmResponse.segments) ? llmResponse.segments : []
  };
}

module.exports = { formatAndValidateOutput };
