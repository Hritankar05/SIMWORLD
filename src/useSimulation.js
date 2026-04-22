/**
 * useSimulation — React hook for WebSocket-based simulation.
 *
 * Connects to the FastAPI WebSocket endpoint and manages real-time
 * tick updates, replacing the old polling/setTimeout tick loop.
 *
 * Usage:
 *   const sim = useSimulation(simulationId);
 *   sim.agents, sim.worldState, sim.events, sim.tick
 *   sim.pause(), sim.resume(), sim.injectEvent(text), sim.changeSpeed(speed)
 */

import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_ATTEMPTS = 10;

export default function useSimulation(simulationId) {
  // ── State ────────────────────────────────────────────────────────
  const [agents, setAgents] = useState([]);
  const [agentStates, setAgentStates] = useState({});
  const [worldState, setWorldState] = useState({ tick: 0, marketIndex: 1000 });
  const [events, setEvents] = useState([]);
  const [tick, setTick] = useState(0);
  const [marketHistory, setMarketHistory] = useState([1000]);
  const [totalInteractions, setTotalInteractions] = useState(0);
  const [thinkingAgents, setThinkingAgents] = useState(new Set());
  const [isConnected, setIsConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  // ── Refs ──────────────────────────────────────────────────────────
  const wsRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef(null);
  const simulationIdRef = useRef(simulationId);

  // Keep ref in sync
  useEffect(() => {
    simulationIdRef.current = simulationId;
  }, [simulationId]);

  // ── WebSocket connection ──────────────────────────────────────────
  const connect = useCallback(() => {
    if (!simulationId) return;

    // Clean up old connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    const url = `${WS_URL}/ws/simulation/${simulationId}`;
    console.log(`[WS] Connecting to ${url}`);

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected');
      setIsConnected(true);
      reconnectAttempts.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleMessage(data);
      } catch (err) {
        console.error('[WS] Failed to parse message:', err);
      }
    };

    ws.onclose = (event) => {
      console.log(`[WS] Disconnected (code=${event.code})`);
      setIsConnected(false);

      // Auto-reconnect if not intentional close
      if (event.code !== 1000 && simulationIdRef.current) {
        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts.current++;
          const delay = RECONNECT_DELAY_MS * reconnectAttempts.current;
          console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
          reconnectTimer.current = setTimeout(connect, delay);
        } else {
          console.error('[WS] Max reconnect attempts reached');
        }
      }
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
    };
  }, [simulationId]);

  // ── Handle incoming messages ──────────────────────────────────────
  const handleMessage = useCallback((data) => {
    const { type } = data;

    switch (type) {
      case 'tick_update': {
        const { tick: newTick, agentUpdates, worldState: ws, newEvents } = data;

        setTick(newTick);

        // Update world state
        if (ws) {
          setWorldState(ws);
          if (ws.marketIndex !== undefined) {
            setMarketHistory(prev => [...prev, ws.marketIndex]);
          }
        }

        // Update agent states
        if (agentUpdates && Array.isArray(agentUpdates)) {
          setAgentStates(prev => {
            const next = { ...prev };
            agentUpdates.forEach(update => {
              const prevHistory = next[update.agentId]?.history || [];
              next[update.agentId] = {
                thought: update.thought || '',
                action: update.action || '',
                emotionalState: update.emotionalState || 'neutral',
                message: update.message || '',
                targetAgent: update.targetAgent || '',
                history: [...prevHistory, {
                  tick: newTick,
                  thought: update.thought,
                  action: update.action,
                  message: update.message,
                }],
              };
            });
            return next;
          });

          // Agent state updates (names + emotions) handled below after comm events

          setTotalInteractions(prev => prev + agentUpdates.length);
        }

        // Add events to feed
        if (newEvents && Array.isArray(newEvents)) {
          const formatted = newEvents.map(ev => {
            // Parse "AgentName (Role): action" format
            const match = ev.match(/^(.+?)\s*\((.+?)\):\s*(.+)$/);
            if (match) {
              return {
                tick: newTick,
                agentName: match[1],
                message: match[3],
                type: 'action',
                color: '#00D4FF',
              };
            }
            return { tick: newTick, agentName: 'SYSTEM', message: ev, type: 'action', color: '#888' };
          });
          setEvents(prev => [...prev, ...formatted]);
        }

        // Also build communication events from agent updates
        if (agentUpdates && Array.isArray(agentUpdates)) {
          const commEvents = agentUpdates
            .filter(u => u.message)
            .map(u => ({
              tick: newTick,
              agentName: u.agentName || u.agentId,
              message: `💬 → ${u.targetAgent || 'everyone'}: "${u.message}"`,
              type: 'communication',
              color: '#00D4FF',
            }));
          if (commEvents.length > 0) {
            setEvents(prev => [...prev, ...commEvents]);
          }
        }

        // Update agent names for display (ensure we have name, not just ID)
        if (agentUpdates && Array.isArray(agentUpdates)) {
          setAgents(prev => prev.map(a => {
            const update = agentUpdates.find(u => u.agentId === String(a.id));
            if (update) {
              return {
                ...a,
                emotionalState: update.emotionalState || a.emotionalState,
                name: update.agentName || a.name,
              };
            }
            return a;
          }));
        }

        setThinkingAgents(new Set());
        break;
      }

      case 'tick_start': {
        // Mark all agents as thinking
        setThinkingAgents(prev => {
          const next = new Set(prev);
          if (data.agents) {
            data.agents.forEach(id => next.add(id));
          }
          return next;
        });
        break;
      }

      case 'simulation_paused':
        setIsPaused(true);
        break;

      case 'simulation_resumed':
        setIsPaused(false);
        break;

      case 'simulation_completed':
        setIsPaused(true);
        break;

      case 'error':
        console.error('[WS] Server error:', data.message);
        break;

      default:
        console.log('[WS] Unknown message type:', type, data);
    }
  }, []);

  // ── Connect/disconnect on simulationId change ─────────────────────
  useEffect(() => {
    if (simulationId) {
      connect();
    }

    return () => {
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmount');
        wsRef.current = null;
      }
    };
  }, [simulationId, connect]);

  // ── Send commands ─────────────────────────────────────────────────
  const sendCommand = useCallback((command) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(command));
    } else {
      console.warn('[WS] Cannot send — not connected');
    }
  }, []);

  const pause = useCallback(() => {
    sendCommand({ type: 'pause' });
    setIsPaused(true);
  }, [sendCommand]);

  const resume = useCallback(() => {
    sendCommand({ type: 'resume' });
    setIsPaused(false);
  }, [sendCommand]);

  const injectEvent = useCallback((eventText) => {
    sendCommand({ type: 'inject_event', event: eventText });
    // Add injected event to local feed immediately
    setEvents(prev => [...prev, {
      tick: tick,
      agentName: '⚡ INJECTED EVENT',
      message: eventText,
      type: 'injected',
      color: '#FFB800',
    }]);
  }, [sendCommand, tick]);

  const changeSpeed = useCallback((speed) => {
    // speed: 'slow', 'normal', 'fast'
    sendCommand({ type: 'speed_change', speed });
  }, [sendCommand]);

  // ── Reset state ───────────────────────────────────────────────────
  const reset = useCallback(() => {
    setAgents([]);
    setAgentStates({});
    setWorldState({ tick: 0, marketIndex: 1000 });
    setEvents([]);
    setTick(0);
    setMarketHistory([1000]);
    setTotalInteractions(0);
    setThinkingAgents(new Set());
    setIsPaused(false);
  }, []);

  return {
    // State
    agents,
    setAgents,
    agentStates,
    worldState,
    events,
    tick,
    marketHistory,
    totalInteractions,
    thinkingAgents,
    isConnected,
    isPaused,

    // Actions
    pause,
    resume,
    injectEvent,
    changeSpeed,
    reset,
  };
}
