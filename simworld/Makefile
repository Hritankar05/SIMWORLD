# ═══════════════════════════════════════════════════════════════════
# SIMWORLD — Multi-Agent Simulation Platform
# ═══════════════════════════════════════════════════════════════════

.PHONY: dev stop migrate seed dashboard train-finance train-all logs setup clean test

# ── Development ───────────────────────────────────────────────────

dev: ## Start all services in development mode
	docker compose up --build

dev-d: ## Start all services in detached mode
	docker compose up --build -d

stop: ## Stop all services
	docker compose down

stop-clean: ## Stop all services and remove volumes
	docker compose down -v

# ── Database ──────────────────────────────────────────────────────

migrate: ## Run Alembic migrations
	cd backend && alembic upgrade head

migration: ## Create a new Alembic migration
	cd backend && alembic revision --autogenerate -m "$(msg)"

seed: ## Seed the database with sample data
	cd backend && python seed.py

# ── Dashboard ─────────────────────────────────────────────────────

dashboard: ## Run the monitoring dashboard (port 8501)
	python dashboard/dashboard.py

# ── Training ──────────────────────────────────────────────────────

train-finance: ## Train the finance domain model
	python training/train.py --category finance

train-corporate: ## Train the corporate domain model
	python training/train.py --category corporate

train-crisis: ## Train the crisis domain model
	python training/train.py --category crisis

train-social: ## Train the social domain model
	python training/train.py --category social

train-all: ## Train all domain models
	python training/train.py --all

evaluate: ## Evaluate all trained models
	python training/evaluate.py --all

# ── Pipeline ──────────────────────────────────────────────────────

collect-logs: ## Collect and display log statistics
	python pipeline/log_collector.py

watch-training: ## Start the training watcher daemon
	python pipeline/training_watcher.py

# ── Logs ──────────────────────────────────────────────────────────

logs: ## Follow backend logs
	docker compose logs -f backend

logs-all: ## Follow all service logs
	docker compose logs -f

logs-celery: ## Follow Celery worker logs
	docker compose logs -f celery_worker

# ── Setup ─────────────────────────────────────────────────────────

setup: ## Initial project setup
	@echo "📦 Setting up SIMWORLD..."
	@cp -n backend/.env.example backend/.env 2>/dev/null || true
	@echo "✅ Created .env from .env.example"
	@echo "🐳 Starting services..."
	docker compose up -d postgres redis
	@echo "⏳ Waiting for databases..."
	sleep 5
	@echo "🗃️  Running migrations..."
	cd backend && pip install -r requirements.txt
	cd backend && alembic upgrade head
	@echo "🌱 Seeding database..."
	cd backend && python seed.py
	@echo "✅ Setup complete! Run 'make dev' to start."

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf data/*/dataset/ 2>/dev/null || true
	@echo "🧹 Cleaned."

# ── Production ────────────────────────────────────────────────────

prod: ## Start production environment
	docker compose -f docker-compose.prod.yml up --build -d

prod-stop: ## Stop production environment
	docker compose -f docker-compose.prod.yml down

# ── Help ──────────────────────────────────────────────────────────

help: ## Show this help message
	@echo "SIMWORLD — Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
