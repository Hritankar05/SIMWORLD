import React, { useState } from 'react';

const EMOTION_EMOJIS = {
  calm: '😌', anxious: '😰', excited: '🤩', fearful: '😨',
  determined: '💪', frustrated: '😤',
};

const TRAIT_COLORS = {
  assertiveness: '#FF6B6B', empathy: '#4ECDC4', rationality: '#45B7D1', creativity: '#F7DC6F',
};

const DEFAULT_AGENT = {
  id: '', name: '', role: '', goal: '',
  personality: { assertiveness: 0.5, empathy: 0.5, rationality: 0.5, creativity: 0.5 },
  emotionalState: 'calm', riskTolerance: 0.5, color: '#00D4FF',
};

function AgentCard({ agent, onUpdate, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({ ...agent, personality: { ...agent.personality } });

  const save = () => { onUpdate(draft); setEditing(false); };
  const cancel = () => { setDraft({ ...agent, personality: { ...agent.personality } }); setEditing(false); };

  const setTrait = (key, val) => setDraft(d => ({ ...d, personality: { ...d.personality, [key]: val } }));

  return (
    <div className="agent-card glass-card">
      <div className="agent-card-header">
        <div className="agent-avatar" style={{ backgroundColor: agent.color, color: '#fff' }}>
          {agent.name?.[0] || '?'}
        </div>
        <div>
          <div className="agent-name">{agent.name}</div>
          <div className="agent-role">{agent.role}</div>
        </div>
      </div>

      {!editing ? (
        <>
          <div className="agent-goal"><span>Goal:</span> {agent.goal}</div>
          {Object.entries(agent.personality).map(([key, val]) => (
            <div className="trait-bar-wrap" key={key}>
              <div className="trait-label">
                <span>{key}</span><span>{(val * 100).toFixed(0)}%</span>
              </div>
              <div className="trait-bar">
                <div className="trait-bar-fill" style={{ width: `${val * 100}%`, background: TRAIT_COLORS[key] }} />
              </div>
            </div>
          ))}
          <div className="emotion-badge">
            {EMOTION_EMOJIS[agent.emotionalState] || '😐'} {agent.emotionalState}
          </div>
          <div className="agent-card-actions">
            <button className="btn btn-sm" onClick={() => setEditing(true)}>✏️ Edit</button>
            <button className="btn btn-sm btn-danger" onClick={onDelete}>🗑️ Delete</button>
          </div>
        </>
      ) : (
        <div className="agent-edit-form">
          <label>Name</label>
          <input value={draft.name} onChange={e => setDraft(d => ({ ...d, name: e.target.value }))} />
          <label>Role</label>
          <input value={draft.role} onChange={e => setDraft(d => ({ ...d, role: e.target.value }))} />
          <label>Goal</label>
          <textarea value={draft.goal} onChange={e => setDraft(d => ({ ...d, goal: e.target.value }))} rows={2} />
          <label>Emotional State</label>
          <select
            value={draft.emotionalState}
            onChange={e => setDraft(d => ({ ...d, emotionalState: e.target.value }))}
            style={{ width: '100%', padding: '8px', borderRadius: '8px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}
          >
            {Object.keys(EMOTION_EMOJIS).map(e => <option key={e} value={e}>{e}</option>)}
          </select>
          {Object.entries(draft.personality).map(([key, val]) => (
            <div key={key}>
              <label>{key}</label>
              <div className="slider-row">
                <input type="range" min="0" max="1" step="0.01" value={val} onChange={e => setTrait(key, parseFloat(e.target.value))} />
                <span>{(val * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
          <label>Risk Tolerance</label>
          <div className="slider-row">
            <input type="range" min="0" max="1" step="0.01" value={draft.riskTolerance} onChange={e => setDraft(d => ({ ...d, riskTolerance: parseFloat(e.target.value) }))} />
            <span>{(draft.riskTolerance * 100).toFixed(0)}%</span>
          </div>
          <label>Color</label>
          <input type="color" value={draft.color} onChange={e => setDraft(d => ({ ...d, color: e.target.value }))} style={{ width: 60, height: 36, padding: 2 }} />
          <div className="agent-card-actions">
            <button className="btn btn-sm btn-success" onClick={save}>✅ Save</button>
            <button className="btn btn-sm" onClick={cancel}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}

function AddAgentForm({ onAdd, onCancel }) {
  const [agent, setAgent] = useState({ ...DEFAULT_AGENT, id: `custom_${Date.now()}` });
  const setTrait = (k, v) => setAgent(a => ({ ...a, personality: { ...a.personality, [k]: v } }));

  const submit = () => {
    if (!agent.name.trim() || !agent.role.trim()) return;
    onAdd({ ...agent, id: `custom_${Date.now()}` });
  };

  return (
    <div className="agent-card glass-card" style={{ borderColor: 'var(--success)', borderStyle: 'solid' }}>
      <div style={{ fontFamily: 'var(--font-display)', fontSize: '0.85rem', color: 'var(--success)', letterSpacing: '2px', marginBottom: 16 }}>
        + NEW AGENT
      </div>
      <div className="agent-edit-form">
        <label>Name *</label>
        <input value={agent.name} onChange={e => setAgent(a => ({ ...a, name: e.target.value }))} placeholder="Agent name" />
        <label>Role *</label>
        <input value={agent.role} onChange={e => setAgent(a => ({ ...a, role: e.target.value }))} placeholder="Their role" />
        <label>Goal</label>
        <textarea value={agent.goal} onChange={e => setAgent(a => ({ ...a, goal: e.target.value }))} placeholder="What they want to achieve" rows={2} />
        {Object.entries(agent.personality).map(([key, val]) => (
          <div key={key}>
            <label>{key}</label>
            <div className="slider-row">
              <input type="range" min="0" max="1" step="0.01" value={val} onChange={e => setTrait(key, parseFloat(e.target.value))} />
              <span>{(val * 100).toFixed(0)}%</span>
            </div>
          </div>
        ))}
        <label>Risk Tolerance</label>
        <div className="slider-row">
          <input type="range" min="0" max="1" step="0.01" value={agent.riskTolerance} onChange={e => setAgent(a => ({ ...a, riskTolerance: parseFloat(e.target.value) }))} />
          <span>{(agent.riskTolerance * 100).toFixed(0)}%</span>
        </div>
        <label>Color</label>
        <input type="color" value={agent.color} onChange={e => setAgent(a => ({ ...a, color: e.target.value }))} style={{ width: 60, height: 36, padding: 2 }} />
        <div className="agent-card-actions">
          <button className="btn btn-sm btn-success" onClick={submit} disabled={!agent.name.trim() || !agent.role.trim()}>✅ Add Agent</button>
          <button className="btn btn-sm" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

export default function ReviewPanel({ agents, setAgents, onLaunch }) {
  const [adding, setAdding] = useState(false);

  const updateAgent = (idx, updated) => {
    const copy = [...agents];
    copy[idx] = updated;
    setAgents(copy);
  };

  const deleteAgent = (idx) => {
    if (agents.length <= 2) return; // keep at least 2
    setAgents(agents.filter((_, i) => i !== idx));
  };

  const addAgent = (agent) => {
    setAgents([...agents, agent]);
    setAdding(false);
  };

  return (
    <div className="review-screen">
      <div className="review-header">
        <h1>AGENT ROSTER</h1>
        <p>Review and customize your simulation agents before launch</p>
      </div>

      <div className="agents-grid">
        {agents.map((agent, idx) => (
          <AgentCard
            key={agent.id}
            agent={agent}
            onUpdate={(updated) => updateAgent(idx, updated)}
            onDelete={() => deleteAgent(idx)}
          />
        ))}
        {adding ? (
          <AddAgentForm onAdd={addAgent} onCancel={() => setAdding(false)} />
        ) : (
          <div className="add-agent-card" onClick={() => setAdding(true)}>
            + ADD CUSTOM AGENT
          </div>
        )}
      </div>

      <div className="review-actions">
        <button id="launch-btn" className="btn btn-lg btn-success" onClick={onLaunch} disabled={agents.length < 2}>
          🚀 LAUNCH SIMULATION
        </button>
      </div>
    </div>
  );
}
