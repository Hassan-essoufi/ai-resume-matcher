.PHONY: up down logs ps restart smoke eval fairness test lint

up:        ## build and start the whole stack (API, UI, Qdrant, Redis)
	docker compose up -d --build --wait

down:      ## stop the stack (data volumes are kept)
	docker compose down

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

restart:
	docker compose restart api ui

smoke:     ## end-to-end test against the running stack
	docker compose exec -T api python scripts/smoke_test.py http://localhost:8000

eval:      ## ranking quality vs baselines, run inside the API container
	docker compose run --rm -v ./eval:/app/eval -e VECTOR_STORE=memory api sh -c "pip install -q --user scikit-learn && python -m eval.evaluate"

fairness:  ## name-swap fairness test
	docker compose run --rm -v ./eval:/app/eval -e VECTOR_STORE=memory api python -m eval.fairness

test:      ## unit tests (local Python, offline)
	pip install -q -r requirements-dev.txt && pytest -q

lint:
	ruff check . --select E,F --ignore E501
