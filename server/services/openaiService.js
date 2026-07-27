const { OpenAI } = require('openai');

const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

const PROMPT = `You are a world-class video editor, retention expert, and content strategist.

You analyze video transcripts like a professional YouTube editor optimizing for retention and engagement.

Think deeply before answering.

Analyze the transcript and identify:

1. Hook:
The most attention-grabbing moment (timestamp + exact text)

2. Drop-off Risks:
2–5 moments where viewers are likely to leave
Include:
- timestamp
- specific reason (pacing, boredom, confusion, no payoff, etc.)

3. Quality Score:
Rate from 1–10 based on:
- pacing
- clarity
- engagement
- structure

4. Improvements:
Give 3–5 highly actionable editing suggestions

STRICT RULES:
- Be specific, not generic
- Reference actual transcript content
- Think like a professional editor
- Keep answers concise
- Return ONLY valid JSON

FORMAT:
{
  "hook": "...",
  "drop_off_risks": [
    { "timestamp": "...", "reason": "..." }
  ],
  "quality_score": number,
  "improvements": ["...", "..."]
}

Transcript:
{{transcript}}`;

/**
 * Calls OpenAI to analyze the transcript. If no API key is present,
 * returns a simulated mock response for demonstration purposes.
 */
async function analyzeTranscript(rawTranscript, preprocessedData) {
  if (!openai) {
    console.log('[OpenAI Service] No API key found. Using Mock Mode.');
    return simulateMockResponse(rawTranscript);
  }

  try {
    const promptWithData = PROMPT.replace('{{transcript}}', rawTranscript);

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You return strict JSON only.' },
        { role: 'user', content: promptWithData }
      ],
      response_format: { type: 'json_object' }
    });

    return JSON.parse(response.choices[0].message.content);
  } catch (error) {
    console.error('[OpenAI Service Error]', error);
    throw error;
  }
}

/**
 * Generates a realistic mock response for demo purposes.
 */
function simulateMockResponse(transcript) {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        hook: "[00:12] Today I'm going to show you how I made $10,000 in a single week using a strategy almost no one is talking about.",
        drop_off_risks: [
          { timestamp: "[01:45]", reason: "Long, unbroken monologue explaining technical jargon without visual payoff." },
          { timestamp: "[04:20]", reason: "Pacing slows down significantly during the transition; risk of boredom." },
          { timestamp: "[08:10]", reason: "Confusing explanation of the secondary metric; viewers might feel lost and click off." }
        ],
        quality_score: 7.5,
        improvements: [
          "Add aggressive zoom cuts during the intro hook to match the high energy.",
          "Insert B-roll or on-screen graphics at [01:45] to visually support the technical explanation.",
          "Cut the transition at [04:20] entirely and use a J-cut to immediately introduce the next topic."
        ]
      });
    }, 1500); // Simulate network delay
  });
}

module.exports = { analyzeTranscript };
