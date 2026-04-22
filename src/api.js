// NVIDIA Nemotron API helper — calls local proxy to bypass CORS
const PROXY_URL = 'http://localhost:3001/api/chat';
const MODEL = 'nvidia/nvidia-nemotron-nano-9b-v2';

export async function callLLM(systemPrompt, userPrompt, maxTokens = 2048) {
  const response = await fetch(PROXY_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: maxTokens,
      temperature: 0.6,
      top_p: 0.95,
      frequency_penalty: 0,
      presence_penalty: 0,
      stream: false,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`API error: ${response.status} — ${err}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

// Parse JSON from LLM response, stripping markdown fences and thinking tags
export function parseJSON(text) {
  let cleaned = text.trim();
  // Strip <think>...</think> reasoning blocks Nemotron may produce
  cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  // Strip markdown code fences
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim();
  }
  // Try to extract JSON array or object if there's surrounding text
  const jsonMatch = cleaned.match(/(\[[\s\S]*\]|\{[\s\S]*\})/);
  if (jsonMatch) {
    cleaned = jsonMatch[1];
  }
  return JSON.parse(cleaned);
}

// Generate agents from scenario
export async function generateAgents(situation) {
  const system = `You are a simulation architect. Given a scenario, generate 4-6 agents as a JSON array.
Each agent object must have exactly these fields:
- id: string (lowercase, no spaces, e.g., "agent_1")
- name: string (realistic full name)
- role: string (their role in the scenario)
- personality: object with keys "assertiveness", "empathy", "rationality", "creativity" (each 0-1 float)
- goal: string (what they're trying to achieve)
- emotionalState: string (one of: "calm", "anxious", "excited", "fearful", "determined", "frustrated")
- riskTolerance: number (0-1 float)
- color: string (hex color like "#FF6B35")

Return ONLY a valid JSON array. No markdown, no explanation, no wrapping.`;

  const text = await callLLM(system, `Scenario: "${situation}"`, 2048);
  return parseJSON(text);
}

// Get agent action for a tick
export async function getAgentAction(agent, tick, worldSummary, recentEvents, otherAgentsContext, talkTo) {
  const talkToLine = talkTo ? `\nYou MUST directly address ${talkTo} in your message this tick.` : '';

  const system = `You are ${agent.name}, a ${agent.role} in a live simulation.
Goal: ${agent.goal}
Personality: assertiveness=${agent.personality.assertiveness}, empathy=${agent.personality.empathy}, rationality=${agent.personality.rationality}, creativity=${agent.personality.creativity}
Risk tolerance: ${agent.riskTolerance} | Emotional state: ${agent.emotionalState}

${otherAgentsContext ? `OTHER AGENTS IN THIS SIMULATION:\n${otherAgentsContext}` : ''}
${talkToLine}

CRITICAL RULES:
1. The "message" field is MANDATORY — it must contain spoken dialogue (1-2 sentences). NEVER leave it empty or null.
2. You must address another agent BY NAME in your message. Talk to them directly.
3. Your message should be a reaction, proposal, question, agreement, disagreement, warning, or negotiation.
4. "targetAgent" must be the name of the agent you're speaking to.

EXAMPLE RESPONSES:
{"thought":"I need to convince Marcus to join my approach","action":"Proposes a joint strategy to Marcus","emotionalState":"determined","message":"Marcus, I think we should combine our resources here. What do you say?","targetAgent":"Marcus Chen","marketImpact":2}
{"thought":"Sarah's plan is too risky, I need to push back","action":"Challenges Sarah's proposal publicly","emotionalState":"frustrated","message":"Sarah, that's way too aggressive. We'll lose everything if the market turns. Let's be more cautious.","targetAgent":"Sarah Williams","marketImpact":-1}

Respond with ONLY valid JSON. No markdown fences, no explanation, no extra text.`;

  const userMsg = `Tick ${tick}. ${worldSummary}. Recent: ${recentEvents}. Speak to another agent now.`;
  const text = await callLLM(system, userMsg, 400);
  return parseJSON(text);
}

// Generate prediction every 10 ticks with cumulative data
export async function generatePrediction(logSummary, currentTick = 10) {
  const system = `You are a predictive analyst AI. You have observed ${currentTick} ticks of a multi-agent simulation. Analyze ALL the simulation data provided (including any injected world events) and predict what happens next.
Return ONLY valid JSON (no markdown, no explanation) with these fields:
- prediction: string (2-3 sentence prediction of what happens in the next 24 hours from tick ${currentTick})
- confidence: number (0-1, your confidence level based on all data so far)
- keyRisks: array of strings (3-4 key risks going forward)
- likelyOutcome: string (1 sentence summary of most likely outcome)
- marketDirection: string (one of: "up", "down", "stable")`;

  const text = await callLLM(system, `Full simulation history through tick ${currentTick}:\n${logSummary}`, 2048);
  return parseJSON(text);
}
