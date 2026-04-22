import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

const EMOTION_EMOJIS = {
  calm: '😌', anxious: '😰', excited: '🤩', fearful: '😨',
  determined: '💪', frustrated: '😤',
};

const MOODS = ['calm', 'anxious', 'excited', 'fearful', 'determined', 'frustrated'];
const MOOD_COLORS = {
  calm: '#4ECDC4', anxious: '#FFB800', excited: '#FF6B6B',
  fearful: '#8B5CF6', determined: '#00D4FF', frustrated: '#FF4466',
};

/* === LEFT PANEL: Agent Status Cards === */
function LeftPanel({ agents, agentStates, thinkingAgents }) {
  const [expanded, setExpanded] = useState(null);

  return (
    <div className="left-panel">
      <div className="left-panel-title">ACTIVE AGENTS</div>
      {agents.map((agent) => {
        const state = agentStates[agent.id];
        const isThinking = thinkingAgents.has(agent.id);
        const isExpanded = expanded === agent.id;

        return (
          <div
            key={agent.id}
            className={`sim-agent-card glass-card ${isExpanded ? 'active' : ''}`}
            onClick={() => setExpanded(isExpanded ? null : agent.id)}
          >
            <div className="sim-agent-header">
              <div
                className={`sim-agent-avatar ${isThinking ? 'thinking' : ''}`}
                style={{ backgroundColor: agent.color, borderColor: agent.color }}
              >
                {agent.name?.[0]}
              </div>
              <div className="sim-agent-info">
                <div className="sim-agent-name">{agent.name}</div>
                <div className="sim-agent-role">{agent.role}</div>
              </div>
            </div>
            <div className="sim-agent-emotion">
              {EMOTION_EMOJIS[state?.emotionalState || agent.emotionalState]}{' '}
              {state?.emotionalState || agent.emotionalState}
            </div>
            {state?.thought && (
              <div className="sim-agent-thought">💭 {state.thought}</div>
            )}
            {state?.action && (
              <div className="sim-agent-action">▸ {state.action}</div>
            )}
            {state?.message && (
              <div className="sim-agent-message" style={{ borderLeftColor: agent.color }}>
                💬 {state.targetAgent ? <span style={{ color: 'var(--primary)', fontWeight: 600 }}>→ {state.targetAgent}: </span> : ''}{state.message}
              </div>
            )}
            {isExpanded && state?.history?.length > 0 && (
              <div className="sim-agent-expanded">
                {state.history.slice().reverse().map((h, i) => (
                  <div className="history-entry" key={i}>
                    <strong style={{ color: 'var(--primary)' }}>T{h.tick}:</strong> {h.action}
                    {h.message && <div style={{ color: 'var(--text-secondary)', fontSize: '0.72rem', marginTop: 2 }}>💬 "{h.message}"</div>}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* === CENTER PANEL: World Event Feed === */
function CenterPanel({ events, onInject }) {
  const [showInject, setShowInject] = useState(false);
  const [injectText, setInjectText] = useState('');

  const handleInject = () => {
    if (injectText.trim()) {
      onInject(injectText.trim());
      setInjectText('');
      setShowInject(false);
    }
  };

  return (
    <div className="center-panel">
      <div className="center-panel-title">WORLD EVENT FEED</div>
      <div className="event-feed">
        {events.slice().reverse().map((ev, i) => (
          <div
            key={i}
            className={`event-entry ${ev.type === 'injected' ? 'event-injected' : ''} ${ev.type === 'communication' ? 'event-communication' : ''}`}
            style={{ borderColor: ev.type === 'injected' || ev.type === 'communication' ? undefined : ev.color || 'var(--border)' }}
          >
            <span className="event-tick">[TICK {ev.tick}]</span>
            {ev.agentName && (
              <span className="event-agent" style={{ color: ev.color }}>{ev.agentName}:</span>
            )}
            <span className="event-message">{ev.message}</span>
          </div>
        ))}
        {events.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-dim)', padding: 40, fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: 2 }}>
            AWAITING FIRST TICK...
          </div>
        )}
      </div>

      <button className="inject-btn" onClick={() => setShowInject(true)}>
        💥 INJECT EVENT
      </button>

      {showInject && (
        <div className="inject-modal-overlay" onClick={() => setShowInject(false)}>
          <div className="inject-modal glass-card" onClick={e => e.stopPropagation()}>
            <h3>💥 INJECT WORLD EVENT</h3>
            <textarea
              value={injectText}
              onChange={e => setInjectText(e.target.value)}
              placeholder="Describe an event that all agents will react to..."
              autoFocus
            />
            <div className="inject-modal-actions">
              <button className="btn btn-sm" onClick={() => setShowInject(false)}>Cancel</button>
              <button className="btn btn-sm btn-warning" onClick={handleInject} disabled={!injectText.trim()}>
                ⚡ Inject
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* === RIGHT PANEL: World Stats === */
function RightPanel({ tick, marketHistory, agents, agentStates, totalInteractions, prediction, isStockSim }) {
  const chartData = marketHistory.map((val, i) => ({ tick: i, value: val }));

  // Mood distribution
  const moodCounts = {};
  MOODS.forEach(m => { moodCounts[m] = 0; });
  agents.forEach(a => {
    const state = agentStates[a.id];
    const mood = state?.emotionalState || a.emotionalState;
    if (moodCounts[mood] !== undefined) moodCounts[mood]++;
  });
  const total = agents.length || 1;

  const marketCurrent = marketHistory[marketHistory.length - 1] || 1000;
  const marketPrev = marketHistory.length > 1 ? marketHistory[marketHistory.length - 2] : marketCurrent;
  const marketDelta = marketCurrent - marketPrev;
  const marketColor = marketDelta >= 0 ? 'var(--success)' : 'var(--danger)';

  return (
    <div className="right-panel">
      <div className="right-panel-title">WORLD STATISTICS</div>

      <div className="stat-card glass-card">
        <div className="stat-label">SIMULATION TICK</div>
        <div className="tick-display">
          <span className="tick-number">{tick}</span>
          <span className="tick-max">/ 50</span>
        </div>
      </div>

      {isStockSim && (
        <div className="stat-card glass-card">
          <div className="stat-label">MARKET INDEX</div>
          <div className="stat-value" style={{ color: marketColor, fontSize: '1.4rem' }}>
            {marketCurrent.toFixed(1)}
            <span style={{ fontSize: '0.75rem', marginLeft: 8 }}>
              {marketDelta >= 0 ? '▲' : '▼'} {Math.abs(marketDelta).toFixed(1)}
            </span>
          </div>
          <div className="market-chart-wrap">
            <ResponsiveContainer width="100%" height={100}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="marketGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00D4FF" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#00D4FF" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey="value" stroke="#00D4FF" strokeWidth={2} fill="url(#marketGradient)" />
                <Tooltip
                  contentStyle={{ background: 'rgba(10,15,30,0.95)', border: '1px solid rgba(0,212,255,0.2)', borderRadius: 8, fontFamily: 'IBM Plex Mono', fontSize: '0.75rem', color: '#E8ECF4' }}
                  labelFormatter={v => `Tick ${v}`}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="stat-card glass-card">
        <div className="stat-label">MOOD DISTRIBUTION</div>
        <div className="mood-distribution">
          {MOODS.map(mood => (
            <div className="mood-row" key={mood}>
              <span className="mood-label">{EMOTION_EMOJIS[mood]} {mood}</span>
              <div className="mood-bar">
                <div
                  className="mood-bar-fill"
                  style={{ width: `${(moodCounts[mood] / total) * 100}%`, background: MOOD_COLORS[mood] }}
                />
              </div>
              <span className="mood-count">{moodCounts[mood]}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="stat-card glass-card">
        <div className="stat-label">TOTAL INTERACTIONS</div>
        <div className="stat-value primary">{totalInteractions}</div>
      </div>

      {prediction && <PredictionCard prediction={prediction} />}
    </div>
  );
}

function PredictionCard({ prediction }) {
  const dirArrow = prediction.marketDirection === 'up' ? '📈' : prediction.marketDirection === 'down' ? '📉' : '➡️';
  const confColor = prediction.confidence >= 0.7 ? 'var(--success)' : prediction.confidence >= 0.4 ? 'var(--warning)' : 'var(--danger)';

  return (
    <div className="prediction-card glass-card">
      <div className="prediction-title">🔮 24-HOUR PREDICTION {prediction.generatedAtTick ? <span style={{ fontSize: '0.7rem', opacity: 0.6, fontFamily: 'var(--font-mono)' }}>(generated at Tick {prediction.generatedAtTick})</span> : null}</div>
      <div className="prediction-text">{prediction.prediction}</div>
      <div className="prediction-meta">
        <span className="prediction-badge" style={{ background: 'var(--primary-dim)', color: 'var(--primary)', border: '1px solid rgba(0,212,255,0.2)' }}>
          {dirArrow} {prediction.marketDirection}
        </span>
        <span className="prediction-badge" style={{ background: confColor + '22', color: confColor, border: `1px solid ${confColor}44` }}>
          {(prediction.confidence * 100).toFixed(0)}% confidence
        </span>
      </div>
      <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: 10, lineHeight: 1.5 }}>
        <strong style={{ color: 'var(--primary)' }}>Likely Outcome:</strong> {prediction.likelyOutcome}
      </div>
      {prediction.keyRisks?.length > 0 && (
        <ul className="prediction-risks">
          {prediction.keyRisks.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      )}
    </div>
  );
}

export default function SimDashboard({
  situation, agents, worldState, events, agentStates, thinkingAgents,
  tick, marketHistory, totalInteractions, prediction,
  phase, tickSpeed, setTickSpeed, onPause, onResume, onStop, onReset, onExport, onInject,
  isStockSim,
}) {
  const isPaused = phase === 'paused';

  return (
    <div className="sim-dashboard">
      {/* Top Bar */}
      <div className="sim-topbar">
        <div className="sim-topbar-left">
          <div className="sim-topbar-title">SIMWORLD</div>
          <div className="sim-topbar-situation" title={situation}>{situation}</div>
        </div>
        <div className="sim-topbar-right">
          <select className="speed-select" value={tickSpeed} onChange={e => setTickSpeed(Number(e.target.value))}>
            <option value={2000}>🐢 Slow (2s)</option>
            <option value={800}>⚡ Normal (0.8s)</option>
            <option value={200}>🚀 Fast (0.2s)</option>
            <option value={100}>💨 Instant</option>
          </select>
          {isPaused ? (
            <button className="btn btn-sm btn-success" onClick={onResume}>▶ Resume</button>
          ) : (
            <button className="btn btn-sm btn-warning" onClick={onPause}>⏸ Pause</button>
          )}
          <button className="btn btn-sm btn-danger" onClick={onStop} title="Stop simulation permanently">⏹ Stop</button>
          <button className="btn btn-sm" onClick={onExport}>📥 Export</button>
          <button className="btn btn-sm" onClick={onReset} style={{ opacity: 0.7 }}>🔄 Reset</button>
        </div>
      </div>

      <LeftPanel agents={agents} agentStates={agentStates} thinkingAgents={thinkingAgents} />
      <CenterPanel events={events} onInject={onInject} />
      <RightPanel
        tick={tick}
        marketHistory={marketHistory}
        agents={agents}
        agentStates={agentStates}
        totalInteractions={totalInteractions}
        prediction={prediction}
        isStockSim={isStockSim}
      />
    </div>
  );
}
