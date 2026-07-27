const { OpenAI } = require('openai');

const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

const PROMPT = `You are a world-class video editor, retention expert, and content strategist.

You analyze video transcripts like a professional YouTube editor optimizing for retention and engagement.

Analyze the transcript and identify:

1. Hook:
The most attention-grabbing moment (timestamp + exact text)

2. Drop-off Risks:
2–5 moments where viewers are likely to leave
Include:
- timestamp
- specific reason (pacing, boredom, confusion, no payoff, etc.)
- explanation (deep ML-style rationale of WHY this causes drop-off)

3. Quality Score:
Rate from 1–10 based on:
- pacing
- clarity
- engagement
- structure

4. Improvements:
Give 3–5 highly actionable editing suggestions

5. Segments:
Break the transcript into 3-5 logical blocks. For each, provide:
- start_time
- end_time
- text_summary
- risk_score (1-10)
- engagement_score (1-10)

STRICT RULES:
- Return ONLY valid JSON

FORMAT:
{
  "hook": "...",
  "drop_off_risks": [
    { "timestamp": "...", "reason": "...", "explanation": "..." }
  ],
  "quality_score": number,
  "improvements": ["...", "..."],
  "segments": [
    { "start_time": "...", "end_time": "...", "text_summary": "...", "risk_score": number, "engagement_score": number }
  ]
}

Transcript:
{{transcript}}`;

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

function simulateMockResponse(transcript) {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        hook: "[00:12] Today I'm going to show you how I made $10,000 in a single week using a strategy almost no one is talking about.",
        drop_off_risks: [
          { 
            timestamp: "[01:45]", 
            reason: "Long, unbroken monologue", 
            explanation: "Lexical density spike coupled with an absence of visual variation creates a high cognitive load, causing casual viewers to churn." 
          },
          { 
            timestamp: "[04:20]", 
            reason: "Pacing slows down", 
            explanation: "The transition introduces a 4-second silence gap and filler words, breaking the established tempo." 
          },
          { 
            timestamp: "[08:10]", 
            reason: "Confusing metric explanation", 
            explanation: "Introduction of jargon without a grounding metaphor. The semantic complexity exceeds the typical retention threshold for this niche." 
          }
        ],
        quality_score: 7.5,
        improvements: [
          "Add aggressive zoom cuts during the intro hook to match the high energy.",
          "Insert B-roll or on-screen graphics at [01:45] to visually support the technical explanation.",
          "Cut the transition at [04:20] entirely and use a J-cut to immediately introduce the next topic."
        ],
        segments: [
          { start_time: "00:00", end_time: "01:40", text_summary: "High-energy introduction and hook delivery.", risk_score: 2, engagement_score: 9 },
          { start_time: "01:41", end_time: "04:19", text_summary: "Deep technical explanation of the strategy.", risk_score: 8, engagement_score: 5 },
          { start_time: "04:20", end_time: "08:00", text_summary: "Case studies and examples.", risk_score: 4, engagement_score: 7 },
          { start_time: "08:01", end_time: "10:00", text_summary: "Secondary metrics and conclusion.", risk_score: 7, engagement_score: 4 }
        ]
      });
    }, 5000);
  });
}

module.exports = { analyzeTranscript };
