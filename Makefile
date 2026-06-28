# Makefile: Standardizes commands for the team. 
# Usage: `make setup`, `make build`, `make run`, `make precompute`, `make test`


.PHONY: setup build run test precompute gameday

setup:
	uv venv .venv
	uv pip install -r requirements.txt

build:
	docker build -t india-runs-hackthon .

precompute:
	python scripts/precompute.py

run:
	python src/pipeline.py

test:
	pytest tests/ -v

gameday: build
	docker run --rm -v $(PWD):/app/host --memory="16g" --memory-swap="16g" --cpus="1.0" india-runs-hackthon sh -c "python run_pipeline.py && cp submission.csv /app/host/"

sandbox: build
	docker run --rm -p 8000:8000 --memory="16g" --memory-swap="16g" --cpus="1.0" india-runs-hackthon uvicorn sandbox.app:app --host 0.0.0.0 --port 8000