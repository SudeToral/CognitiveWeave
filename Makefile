SHELL        := /bin/bash
SERVER       := 100.98.163.58
SERVER_REPO  := ~/CognitiveWeave
SSH_CMD      := ssh $(SERVER)
SERVER_ENV   := DOCKER_HOST=unix:///Users/suderemote/.colima/default/docker.sock PATH=/opt/homebrew/bin:$$PATH

# ── Lokal ──────────────────────────────────────────────────────────────────
.PHONY: app
app:
	uv run streamlit run app.py

.PHONY: test
test:
	uv run pytest tests/ -q

.PHONY: lint
lint:
	uv run ruff check src/ tests/ && uv run mypy src/

# ── Server ─────────────────────────────────────────────────────────────────
.PHONY: deploy
deploy:
	$(SSH_CMD) "cd $(SERVER_REPO) && git pull"

.PHONY: up
up:
	$(SSH_CMD) "cd $(SERVER_REPO) && env $(SERVER_ENV) docker-compose up -d"

.PHONY: down
down:
	$(SSH_CMD) "cd $(SERVER_REPO) && env $(SERVER_ENV) docker-compose down"

.PHONY: ps
ps:
	$(SSH_CMD) "env $(SERVER_ENV) docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"

.PHONY: logs
logs:
	$(SSH_CMD) "cd $(SERVER_REPO) && env $(SERVER_ENV) docker-compose logs --tail=50"

.PHONY: restart
restart: deploy up
