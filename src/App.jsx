import React, { useState, useRef, useCallback, useEffect } from 'react';
import InputScreen from './InputScreen.jsx';
import ReviewPanel from './ReviewPanel.jsx';
import SimDashboard from './SimDashboard.jsx';
import {
  createSimulation,
  startSimulation,
  pauseSimulation as apiPause,
  resumeSimulation as apiResume,
  stopSimulation as apiStop,
  getPrediction,
  checkHealth,
} from './client.js';
import useSimulation from './useSimulation.js';
// Keep the old api.js as fallback when backend is offline
import { generateAgents, getAgentAction, generatePrediction } from './api.js';

const MAX_TICKS = 50;
const PREDICTION_TICK = 10;

function isStockSimulation(situation) {
  const keywords = ['stock', 'market', 'trading', 'wall street', 'nyse', 'nasdaq', 'shares', 'portfolio', 'investor', 'hedge fund', 'bull market', 'bear market', 'earnings', 'ipo', 'dow jones', 'forex', 'commodity', 'bitcoin', 'crypto'];
  const lower = (situation || '').toLowerCase();
  return keywords.some(k => lower.includes(k));
}

function buildWorldSummary(worldState, agents, agentStates, isStockSim) {
  const items = agents.map(a => {
    const s = agentStates[a.id];
    return s ? `${a.name} (${a.role}, feeling ${s.emotionalState}): last did "${s.action}"` : `${a.name} (${a.role})`;
  });
  const marketStr = isStockSim ? ` Market index: ${worldState.marketIndex.toFixed(1)}.` : '';
  return `Tick ${worldState.tick}.${marketStr} Agents: ${items.join('; ')}`;
}

function buildRecentEvents(events, count = 5) {
  const recent = events.slice(-count);
  if (recent.length === 0) return 'No events yet.';
  return recent.map(e => `[T${e.tick}] ${e.agentName || 'SYSTEM'}: ${e.message}`).join(' | ');
}

export default function App() {
  // ── Backend mode detection ──────────────────────────────────────
  const [backendOnline, setBackendOnline] = useState(false);
  const [simulationId, setSimulationId] = useState(null);

  // Check backend on mount
  useEffect(() => {
    checkHealth().then(h => {
      setBackendOnline(h.online);
      console.log(`Backend: ${h.online ? '🟢 online' : '🔴 offline (using fallback)'}`);
    });
  }, []);

  // ── WebSocket hook (only active when backend online + simulation running)
  const ws = useSimulation(backendOnline ? simulationId : null);

  // ── Core state ──────────────────────────────────────────────────
  const [phase, setPhase] = useState('input');
  const [situation, setSituation] = useState('');
  const [agents, setAgents] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [error, setError] = useState('');

  // ── Fallback (offline) state ────────────────────────────────────
  const [tick, setTick] = useState(0);
  const [events, setEvents] = useState([]);
  const [agentStates, setAgentStates] = useState({});
  const [marketIndex, setMarketIndex] = useState(1000);
  const [marketHistory, setMarketHistory] = useState([1000]);
  const [injectedEvents, setInjectedEvents] = useState([]);
  const [totalInteractions, setTotalInteractions] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [tickSpeed, setTickSpeed] = useState(800);
  const [thinkingAgents, setThinkingAgents] = useState(new Set());
  const [simulationLog, setSimulationLog] = useState([]);

  const tickRef = useRef(null);
  const stateRef = useRef({});
  const isRunningRef = useRef(false);
  const injectedEventsRef = useRef([]);

  // Keep ref in sync (fallback mode)
  useEffect(() => {
    stateRef.current = { tick, events, agentStates, marketIndex, marketHistory, injectedEvents, totalInteractions, agents, situation, simulationLog };
  });

  // ── Resolve which state to use (WS vs fallback) ─────────────────
  const activeAgents = backendOnline && simulationId ? (ws.agents.length > 0 ? ws.agents : agents) : agents;
  const activeAgentStates = backendOnline && simulationId ? ws.agentStates : agentStates;
  const activeTick = backendOnline && simulationId ? ws.tick : tick;
  const activeEvents = backendOnline && simulationId ? ws.events : events;
  const activeMarketHistory = backendOnline && simulationId ? ws.marketHistory : marketHistory;
  const activeTotalInteractions = backendOnline && simulationId ? ws.totalInteractions : totalInteractions;
  const activeThinkingAgents = backendOnline && simulationId ? ws.thinkingAgents : thinkingAgents;
  const activeMarketIndex = backendOnline && simulationId
    ? (ws.worldState.marketIndex || activeMarketHistory[activeMarketHistory.length - 1] || 1000)
    : marketIndex;

  // ── Fetch prediction from backend when tick hits intervals ──────
  useEffect(() => {
    if (backendOnline && simulationId && activeTick > 0 && activeTick % PREDICTION_TICK === 0) {
      getPrediction(simulationId)
        .then(pred => setPrediction({ ...pred, generatedAtTick: activeTick }))
        .catch(e => console.warn('Prediction fetch failed:', e));
    }
  }, [backendOnline, simulationId, activeTick]);

  // ══════════════════════════════════════════════════════════════════
  // GENERATE AGENTS
  // ══════════════════════════════════════════════════════════════════
  const handleGenerate = useCallback(async () => {
    setError('');
    setIsLoading(true);
    setLoadingMsg('Initializing simulation architect...');

    try {
      if (backendOnline) {
        // ── Backend mode: create simulation (agents generated server-side)
        setLoadingMsg('Creating simulation on backend...');
        const sim = await createSimulation(situation);
        // Backend returns simulation_id (not id)
        const simId = sim.simulation_id || sim.id;
        setSimulationId(simId);
        // Remap snake_case agent fields from API to camelCase used by frontend
        const mappedAgents = (sim.agents || []).map(a => ({
          ...a,
          emotionalState: a.emotionalState || a.emotional_state || 'calm',
          riskTolerance: a.riskTolerance || a.risk_tolerance || 0.5,
          createdBy: a.createdBy || a.created_by || 'auto',
        }));
        setAgents(mappedAgents);
        // Also push agents into WS hook
        ws.setAgents(mappedAgents);
      } else {
        // ── Fallback: use old API
        const generated = await generateAgents(situation);
        setAgents(generated);
      }
      setPhase('review');
    } catch (err) {
      setError(`Failed to generate agents: ${err.message}`);
    } finally {
      setIsLoading(false);
      setLoadingMsg('');
    }
  }, [situation, backendOnline, ws]);

  // ══════════════════════════════════════════════════════════════════
  // BUILD AGENT CONTEXT (fallback mode)
  // ══════════════════════════════════════════════════════════════════
  const buildAgentContext = useCallback((agent, agents, agentStates) => {
    const others = agents.filter(a => a.id !== agent.id);
    if (others.length === 0) return '';
    return others.map(a => {
      const st = agentStates[a.id];
      if (st) {
        const msg = st.message ? ` Last said: "${st.message}"` : '';
        return `- ${a.name} (${a.role}, feeling ${st.emotionalState}): Last action: "${st.action}".${msg}`;
      }
      return `- ${a.name} (${a.role}, feeling ${a.emotionalState}): No actions yet.`;
    }).join('\n');
  }, []);

  // ══════════════════════════════════════════════════════════════════
  // TICK LOOP (fallback mode only — when backend is offline)
  // ══════════════════════════════════════════════════════════════════
  const runTick = useCallback(async () => {
    if (!isRunningRef.current) return;
    const s = stateRef.current;
    const currentTick = s.tick + 1;

    if (currentTick > MAX_TICKS) {
      isRunningRef.current = false;
      setPhase('paused');
      return;
    }

    const isStockSim = isStockSimulation(s.situation);
    const worldSummary = buildWorldSummary(
      { tick: currentTick, marketIndex: s.marketIndex }, s.agents, s.agentStates, isStockSim
    );

    const pendingInjected = [...injectedEventsRef.current];
    injectedEventsRef.current = [];

    let recentStr = buildRecentEvents(s.events);
    if (pendingInjected.length > 0) {
      recentStr += ' | ⚡ BREAKING WORLD EVENT: ' + pendingInjected.join('; ') + ' — All agents must react.';
    }

    setThinkingAgents(new Set(s.agents.map(a => a.id)));

    const agentTargets = s.agents.map((agent, idx) => {
      const others = s.agents.filter(a => a.id !== agent.id);
      const targetIdx = (currentTick + idx) % others.length;
      return others[targetIdx]?.name || others[0]?.name || '';
    });

    const results = await Promise.allSettled(
      s.agents.map((agent, idx) => {
        const otherAgentsContext = buildAgentContext(agent, s.agents, s.agentStates);
        return getAgentAction(agent, currentTick, worldSummary, recentStr, otherAgentsContext, agentTargets[idx]);
      })
    );

    if (!isRunningRef.current) return;

    const newEvents = [];
    const newAgentStates = { ...s.agentStates };
    let newMarket = s.marketIndex;
    let newInteractions = s.totalInteractions;

    pendingInjected.forEach(ie => {
      newEvents.push({ tick: currentTick, type: 'injected', message: ie, agentName: '⚡ INJECTED EVENT', color: '#FFB800' });
    });

    s.agents.forEach((agent, i) => {
      const result = results[i];
      if (result.status === 'fulfilled') {
        const data = result.value;
        const prevHistory = newAgentStates[agent.id]?.history || [];

        let message = (data.message && data.message.trim()) ? data.message.trim() : '';
        let targetAgent = (data.targetAgent && data.targetAgent.trim()) ? data.targetAgent.trim() : agentTargets[i];

        if (!message) {
          const action = data.action || 'observing the situation';
          message = `${targetAgent}, I'm ${action.toLowerCase().replace(/^(he |she |they |i )/, '')}. What's your take on this?`;
        }

        newAgentStates[agent.id] = {
          thought: data.thought || '',
          action: data.action || '',
          emotionalState: data.emotionalState || agent.emotionalState,
          message, targetAgent,
          history: [...prevHistory, { tick: currentTick, thought: data.thought, action: data.action, message }],
        };

        const impact = typeof data.marketImpact === 'number' ? data.marketImpact : 0;
        newMarket += impact;

        newEvents.push({ tick: currentTick, agentName: agent.name, message: data.action || '', color: agent.color, type: 'action' });
        newEvents.push({ tick: currentTick, agentName: agent.name, message: `💬 → ${targetAgent}: "${message}"`, color: agent.color, type: 'communication' });
        newInteractions++;
      } else {
        newEvents.push({ tick: currentTick, agentName: agent.name, message: '[Communication lost this tick]', color: '#555', type: 'error' });
      }
    });

    newMarket = Math.max(0, newMarket);
    const logEntry = { tick: currentTick, agentActions: { ...newAgentStates }, marketIndex: newMarket, events: newEvents };

    setTick(currentTick);
    setEvents(prev => [...prev, ...newEvents]);
    setAgentStates(newAgentStates);
    setMarketIndex(newMarket);
    setMarketHistory(prev => [...prev, newMarket]);
    if (pendingInjected.length > 0) setInjectedEvents([]);
    setTotalInteractions(newInteractions);
    setSimulationLog(prev => [...prev, logEntry]);
    setThinkingAgents(new Set());

    setAgents(prev => prev.map(a => {
      const st = newAgentStates[a.id];
      return st ? { ...a, emotionalState: st.emotionalState } : a;
    }));

    if (currentTick > 0 && currentTick % PREDICTION_TICK === 0) {
      const allEvents = [...s.events, ...newEvents];
      const logSummary = allEvents.map(e => `[T${e.tick}] ${e.agentName}: ${e.message}`).join('\n');
      generatePrediction(logSummary, currentTick)
        .then(pred => setPrediction({ ...pred, generatedAtTick: currentTick }))
        .catch(e => console.error('Prediction failed:', e));
    }

    if (currentTick >= MAX_TICKS) {
      isRunningRef.current = false;
      setPhase('paused');
    }
  }, [buildAgentContext]);

  // Fallback tick loop effect
  useEffect(() => {
    if (!backendOnline && phase === 'simulation' && isRunningRef.current) {
      const loop = async () => {
        if (!isRunningRef.current) return;
        await runTick();
        if (isRunningRef.current) {
          tickRef.current = setTimeout(loop, stateRef.current.tick >= MAX_TICKS ? 0 : Math.max(tickSpeed, 100));
        }
      };
      tickRef.current = setTimeout(loop, 200);
    }
    return () => { if (tickRef.current) clearTimeout(tickRef.current); };
  }, [phase, tickSpeed, runTick, backendOnline]);

  // ══════════════════════════════════════════════════════════════════
  // LAUNCH
  // ══════════════════════════════════════════════════════════════════
  const handleLaunch = useCallback(async () => {
    if (backendOnline && simulationId) {
      try {
        await startSimulation(simulationId);
      } catch (err) {
        console.error('Failed to start simulation:', err);
      }
    } else {
      isRunningRef.current = true;
    }
    setPhase('simulation');
  }, [backendOnline, simulationId]);

  // ══════════════════════════════════════════════════════════════════
  // PAUSE / RESUME
  // ══════════════════════════════════════════════════════════════════
  const handlePause = useCallback(async () => {
    if (backendOnline && simulationId) {
      try {
        await apiPause(simulationId);
        ws.pause();
      } catch (err) {
        console.error('Pause failed:', err);
      }
    } else {
      isRunningRef.current = false;
      if (tickRef.current) clearTimeout(tickRef.current);
    }
    setPhase('paused');
  }, [backendOnline, simulationId, ws]);

  const handleResume = useCallback(async () => {
    const currentTick = backendOnline && simulationId ? activeTick : tick;
    if (currentTick >= MAX_TICKS) return;

    if (backendOnline && simulationId) {
      try {
        await apiResume(simulationId);
        ws.resume();
      } catch (err) {
        console.error('Resume failed:', err);
      }
    } else {
      isRunningRef.current = true;
    }
    setPhase('simulation');
  }, [backendOnline, simulationId, ws, tick, activeTick]);

  // ══════════════════════════════════════════════════════════════════
  // STOP
  // ══════════════════════════════════════════════════════════════════
  const handleStop = useCallback(async () => {
    if (backendOnline && simulationId) {
      try {
        await apiStop(simulationId);
      } catch (err) {
        console.error('Stop failed:', err);
      }
    }
    isRunningRef.current = false;
    if (tickRef.current) clearTimeout(tickRef.current);
    setPhase('paused');
  }, [backendOnline, simulationId]);

  // ══════════════════════════════════════════════════════════════════
  // RESET
  // ══════════════════════════════════════════════════════════════════
  const handleReset = useCallback(() => {
    isRunningRef.current = false;
    if (tickRef.current) clearTimeout(tickRef.current);
    ws.reset();

    setPhase('input');
    setSituation('');
    setAgents([]);
    setSimulationId(null);
    setTick(0);
    setEvents([]);
    setAgentStates({});
    setMarketIndex(1000);
    setMarketHistory([1000]);
    setInjectedEvents([]);
    setTotalInteractions(0);
    setPrediction(null);
    setSimulationLog([]);
    setThinkingAgents(new Set());
  }, [ws]);

  // ══════════════════════════════════════════════════════════════════
  // INJECT EVENT
  // ══════════════════════════════════════════════════════════════════
  const handleInject = useCallback((eventText) => {
    if (backendOnline && simulationId) {
      ws.injectEvent(eventText);
    } else {
      injectedEventsRef.current = [...injectedEventsRef.current, eventText];
      setInjectedEvents(prev => [...prev, eventText]);
    }
  }, [backendOnline, simulationId, ws]);

  // ══════════════════════════════════════════════════════════════════
  // SPEED CHANGE
  // ══════════════════════════════════════════════════════════════════
  const handleSpeedChange = useCallback((speed) => {
    setTickSpeed(speed);
    if (backendOnline && simulationId) {
      const speedMap = { 2000: 'slow', 800: 'normal', 200: 'fast', 100: 'fast' };
      ws.changeSpeed(speedMap[speed] || 'normal');
    }
  }, [backendOnline, simulationId, ws]);

  // ══════════════════════════════════════════════════════════════════
  // EXPORT
  // ══════════════════════════════════════════════════════════════════
  const handleExport = useCallback(() => {
    const blob = new Blob([JSON.stringify({
      situation,
      agents: activeAgents,
      simulationLog,
      prediction,
      totalTicks: activeTick,
      finalMarketIndex: activeMarketIndex,
      backend: backendOnline ? 'connected' : 'offline',
      simulationId,
    }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `simworld_log_${Date.now()}.json`; a.click();
    URL.revokeObjectURL(url);
  }, [situation, activeAgents, simulationLog, prediction, activeTick, activeMarketIndex, backendOnline, simulationId]);

  // ══════════════════════════════════════════════════════════════════
  // RENDER
  // ══════════════════════════════════════════════════════════════════
  return (
    <>
      {/* Backend status indicator */}
      <div style={{
        position: 'fixed', bottom: 12, right: 12, zIndex: 500,
        padding: '4px 10px', borderRadius: 6, fontSize: '0.7rem',
        fontFamily: 'var(--font-mono)', letterSpacing: 0.5,
        background: backendOnline ? 'rgba(16,185,129,0.15)' : 'rgba(244,63,94,0.15)',
        color: backendOnline ? '#10b981' : '#f43f5e',
        border: `1px solid ${backendOnline ? 'rgba(16,185,129,0.3)' : 'rgba(244,63,94,0.3)'}`,
      }}>
        {backendOnline ? '🟢 Backend' : '🔴 Offline'}{ws.isConnected ? ' · WS' : ''}
      </div>

      {/* Error toast */}
      {error && (
        <div style={{
          position: 'fixed', top: 16, left: '50%', transform: 'translateX(-50%)', zIndex: 500,
          padding: '10px 24px', borderRadius: 8, background: 'var(--danger-dim)',
          border: '1px solid var(--danger)', color: 'var(--danger)',
          fontFamily: 'var(--font-mono)', fontSize: '0.8rem', maxWidth: 600,
          cursor: 'pointer',
        }} onClick={() => setError('')}>
          ⚠ {error} <span style={{ opacity: 0.5, marginLeft: 10 }}>(click to dismiss)</span>
        </div>
      )}

      {/* Loading overlay */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <div className="loading-text">{loadingMsg || 'PROCESSING...'}</div>
        </div>
      )}

      {/* Phase-based rendering */}
      {phase === 'input' && (
        <InputScreen
          situation={situation}
          setSituation={setSituation}
          onGenerate={handleGenerate}
          isLoading={isLoading}
        />
      )}

      {phase === 'review' && (
        <ReviewPanel
          agents={activeAgents}
          setAgents={setAgents}
          onLaunch={handleLaunch}
        />
      )}

      {(phase === 'simulation' || phase === 'paused') && (
        <SimDashboard
          situation={situation}
          agents={activeAgents}
          worldState={{ tick: activeTick, marketIndex: activeMarketIndex }}
          events={activeEvents}
          agentStates={activeAgentStates}
          thinkingAgents={activeThinkingAgents}
          tick={activeTick}
          marketHistory={activeMarketHistory}
          totalInteractions={activeTotalInteractions}
          prediction={prediction}
          phase={phase}
          tickSpeed={tickSpeed}
          setTickSpeed={handleSpeedChange}
          onPause={handlePause}
          onResume={handleResume}
          onStop={handleStop}
          onReset={handleReset}
          onExport={handleExport}
          onInject={handleInject}
          isStockSim={isStockSimulation(situation)}
        />
      )}
    </>
  );
}
