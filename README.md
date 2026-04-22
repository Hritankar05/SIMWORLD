# ⚡ SIMWORLD — Multi-Agent Simulation Platform

A production-grade backend for running multi-agent simulations powered by NVIDIA NIM (Nemotron) models. Agents autonomously reason, act, and interact within dynamic scenarios — generating training data that progressively fine-tunes domain-specific models.

## Architecture

```
┌─────────────┐    ┌──────────┐    ┌───────────┐
│   Frontend   │◄──►│ FastAPI  │◄──►│ PostgreSQL │
│  (React/WS)  │    │ Backend  │    │  Database  │
└─────────────┘    └────┬─────┘    └───────────┘
                        │
                   ┌────┴─────┐
                   │  Redis   │
                   │ Pub/Sub  │
                   └────┬─────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  NVIDIA  │  │ Training │  │  Celery  │
    │   NIM    │  │ Pipeline │  │  Worker  │
    └──────────┘  └──────────┘  └──────────┘
```

## Features

- **Multi-Agent Simulation**: 4-6 AI agents per scenario with unique personalities and goals
- **NVIDIA NIM Integration**: Uses Nemotron models for agent reasoning, classification, and prediction
- **Real-time WebSocket**: Live tick updates streamed to connected clients
- **Automated Training Pipeline**: Simulation data auto-feeds into fine-tuning datasets
- **Situation Classification**: Keywords + LLM classifier routes data to domain-specific categories
- **Outcome Prediction**: After 10+ ticks, predict 24-hour simulation outcomes
- **Model Registry**: Track fine-tuned model quality and auto-graduate when threshold met
- **Dashboard**: Real-time training status monitoring

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- NVIDIA API Key ([get one here](https://build.nvidia.com/))

### Setup

```bash
# 1. Clone and enter the directory
cd simworld

# 2. Copy environment file and add your NVIDIA API key
cp backend/.env.example backend/.env
# Edit backend/.env and set NVIDIA_API_KEY=nvapi-your-key

# 3. Run the automated setup
make setup

# 4. Start everything
make dev
```

The API is now running at `http://localhost:8000`.  
API docs at `http://localhost:8000/docs`.

### Manual Setup (without Docker)

```bash
# Start PostgreSQL and Redis locally, then:
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials

# Run migrations
alembic upgrade head

# Seed sample data
python seed.py

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Reference

### Simulations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/simulations` | Create a new simulation |
| `POST` | `/api/simulations/{id}/start` | Start the tick loop |
| `POST` | `/api/simulations/{id}/pause` | Pause simulation |
| `POST` | `/api/simulations/{id}/resume` | Resume simulation |
| `GET`  | `/api/simulations/{id}` | Get simulation details |
| `GET`  | `/api/simulations/{id}/logs` | Get tick logs |
| `GET`  | `/api/simulations/{id}/prediction` | Get 24h prediction (≥10 ticks) |

### Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/agents/{id}` | Get agent details |
| `PATCH`| `/api/agents/{id}` | Update agent properties |
| `GET`  | `/api/agents/simulation/{sim_id}` | List agents for simulation |

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/training/status` | Training status per category |

### WebSocket

Connect to `ws://localhost:8000/ws/simulation/{simulation_id}`

**Server pushes:**
```json
{
  "type": "tick_update",
  "tick": 5,
  "agentUpdates": [
    {
      "agentId": "uuid",
      "thought": "The market is crashing...",
      "action": "sell_position",
      "emotionalState": "anxious",
      "message": "I'm liquidating my position."
    }
  ],
  "worldState": {"marketIndex": -15, "events": [...], "tick": 5},
  "newEvents": ["Marcus (Trader): sell_position"]
}
```

**Client commands:**
```json
{"type": "inject_event", "event": "Breaking: SEC announces fraud charges"}
{"type": "pause"}
{"type": "resume"}
{"type": "speed_change", "speed": "fast"}
```

## Project Structure

```
simworld/
├── backend/          # FastAPI application
│   ├── api/          # REST + WebSocket routes
│   ├── core/         # Config, database, Redis
│   ├── models/       # SQLAlchemy ORM models
│   ├── schemas/      # Pydantic request/response schemas
│   ├── services/     # Business logic (engine, classifier, etc.)
│   ├── ml/           # Model registry, inference routing
│   └── workers/      # Celery background tasks
├── pipeline/         # Standalone data pipeline
├── training/         # Model training scripts
├── dashboard/        # Monitoring UI
├── data/             # Training data (JSONL per category)
└── models/           # Fine-tuned model artifacts
```

## Training Pipeline

The system automatically collects simulation data and triggers training:

1. **Data Collection**: Every tick generates a training example
2. **Routing**: Log router sends data to category-specific JSONL files
3. **Threshold**: When a category reaches 100 records, training is triggered
4. **Training**: LoRA fine-tuning on the category dataset
5. **Evaluation**: Quality score computed on held-out validation set
6. **Graduation**: Models scoring ≥ 0.80 are activated for inference

```bash
# Manual training
make train-finance      # Train finance model
make train-all          # Train all categories
make evaluate           # Evaluate all models
```

## Dashboard

```bash
make dashboard
# Opens at http://localhost:8501
```

## Make Commands

```bash
make help              # Show all commands
make dev               # Start dev environment
make stop              # Stop services
make migrate           # Run DB migrations
make seed              # Seed sample data
make logs              # Follow backend logs
make train-all         # Train all models
make prod              # Start production
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | — | NVIDIA NIM API key (required) |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async PostgreSQL URL |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `TICK_SPEED_MS` | `1500` | Milliseconds between ticks |
| `TRAINING_THRESHOLD` | `100` | Records before auto-training |
| `MODEL_QUALITY_THRESHOLD` | `0.80` | Score for model graduation |

## License

MIT
