.PHONY: test bot worker run install

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

bot:
	python -m music_dna_agent.src.bot

worker:
	python -m music_dna_agent.src.worker.pipeline_worker

run:
	@echo "Starting Music DNA Background Worker & Telegram Bot..."
	python -m music_dna_agent.src.worker.pipeline_worker & python -m music_dna_agent.src.bot
