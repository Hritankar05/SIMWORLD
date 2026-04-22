import React from 'react';

const EXAMPLES = [
  '🏦 Stock market crash during earnings season',
  '🚀 Startup on launch day with server issues',
  '🌃 City during a massive blackout',
  '🏥 Hospital during a pandemic surge',
  '⚖️ Courtroom during a high-profile trial',
  '🛳️ Cruise ship with an onboard mystery',
  '🏫 University during final exam week chaos',
  '🌪️ Small town facing a Category 5 hurricane',
];

export default function InputScreen({ situation, setSituation, onGenerate, isLoading }) {
  return (
    <div className="input-screen">
      <h1 className="hero-title">SIMWORLD</h1>
      <p className="hero-subtitle">Multi-Agent Simulated World</p>

      <div className="scenario-box">
        <textarea
          id="scenario-input"
          className="scenario-textarea"
          placeholder="Describe any real-world scenario and watch intelligent agents simulate it in real-time..."
          value={situation}
          onChange={(e) => setSituation(e.target.value)}
          disabled={isLoading}
        />
        <div className="generate-btn-wrap">
          <button
            id="generate-btn"
            className="btn btn-lg"
            onClick={onGenerate}
            disabled={!situation.trim() || isLoading}
          >
            {isLoading ? '⏳ Generating Agents...' : '⚡ Generate Simulation'}
          </button>
        </div>
      </div>

      <div className="examples-section">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="example-chip"
            onClick={() => setSituation(ex)}
            disabled={isLoading}
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
