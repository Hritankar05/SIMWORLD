/**
 * SIMWORLD API Client — connects React frontend to FastAPI backend.
 *
 * Replaces direct NVIDIA NIM API calls with backend endpoints.
 * Falls back to the old proxy-based API if backend is unreachable.
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Base fetch wrapper with error handling.
 */
async function apiFetch(path, options = {}) {
  const url = `${API_URL}${path}`;
  const config = {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  };

  const response = await fetch(url, config);

  if (!response.ok) {
    let errMsg;
    try {
      const errData = await response.json();
      errMsg = errData.detail || errData.error || JSON.stringify(errData);
    } catch {
      errMsg = await response.text();
    }
    throw new Error(`API ${response.status}: ${errMsg}`);
  }

  return response.json();
}

// ════════════════════════════════════════════════════════════════════
// Simulations
// ════════════════════════════════════════════════════════════════════

/**
 * Create a new simulation. The backend generates agents automatically.
 * @param {string} situation - The scenario description.
 * @param {Array|null} customAgents - Optional pre-defined agents.
 * @returns {Promise<Object>} - { id, situation, category, status, agents, ... }
 */
export async function createSimulation(situation, customAgents = null) {
  const body = { situation };
  if (customAgents && customAgents.length > 0) {
    body.custom_agents = customAgents;
  }
  return apiFetch('/api/simulations', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/**
 * Start the simulation tick loop on the backend.
 */
export async function startSimulation(id) {
  return apiFetch(`/api/simulations/${id}/start`, { method: 'POST' });
}

/**
 * Pause a running simulation.
 */
export async function pauseSimulation(id) {
  return apiFetch(`/api/simulations/${id}/pause`, { method: 'POST' });
}

/**
 * Resume a paused simulation.
 */
export async function resumeSimulation(id) {
  return apiFetch(`/api/simulations/${id}/resume`, { method: 'POST' });
}

/**
 * Stop a simulation permanently.
 */
export async function stopSimulation(id) {
  return apiFetch(`/api/simulations/${id}/stop`, { method: 'POST' });
}

/**
 * Get simulation details including agents.
 */
export async function getSimulation(id) {
  return apiFetch(`/api/simulations/${id}`);
}

/**
 * Get paginated tick logs.
 */
export async function getSimulationLogs(id, limit = 100, offset = 0) {
  return apiFetch(`/api/simulations/${id}/logs?limit=${limit}&offset=${offset}`);
}

/**
 * Get 24-hour prediction (requires >= 10 ticks).
 */
export async function getPrediction(id) {
  return apiFetch(`/api/simulations/${id}/prediction`);
}

// ════════════════════════════════════════════════════════════════════
// Agents
// ════════════════════════════════════════════════════════════════════

/**
 * Get a single agent's details.
 */
export async function getAgent(id) {
  return apiFetch(`/api/agents/${id}`);
}

/**
 * Update an agent's properties (name, role, goal, etc).
 */
export async function updateAgent(id, updates) {
  return apiFetch(`/api/agents/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });
}

/**
 * List all agents in a simulation.
 */
export async function getAgentsBySimulation(simulationId) {
  return apiFetch(`/api/agents/simulation/${simulationId}`);
}

// ════════════════════════════════════════════════════════════════════
// Training
// ════════════════════════════════════════════════════════════════════

/**
 * Get training status for all categories.
 */
export async function getTrainingStatus() {
  return apiFetch('/api/training/status');
}

// ════════════════════════════════════════════════════════════════════
// Health
// ════════════════════════════════════════════════════════════════════

/**
 * Check if the backend is reachable.
 */
export async function checkHealth() {
  try {
    const data = await apiFetch('/health');
    return { online: true, ...data };
  } catch {
    return { online: false, status: 'unreachable' };
  }
}
