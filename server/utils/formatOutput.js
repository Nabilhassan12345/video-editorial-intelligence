/**
 * Validates and ensures the LLM output conforms strictly to the expected JSON schema.
 */
function formatAndValidateOutput(llmResponse) {
  // Basic validation to ensure required fields exist, fallback to defaults if not
  return {
    hook: llmResponse.hook || "No specific hook identified.",
    drop_off_risks: Array.isArray(llmResponse.drop_off_risks) ? llmResponse.drop_off_risks : [],
    quality_score: typeof llmResponse.quality_score === 'number' ? llmResponse.quality_score : 5,
    improvements: Array.isArray(llmResponse.improvements) ? llmResponse.improvements : []
  };
}

module.exports = { formatAndValidateOutput };
