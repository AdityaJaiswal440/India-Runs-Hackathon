.PHONY: setup build run test gameday

setup:
	uv venv .venv
	uv pip install -r requirements.txt

build:
	docker build -t india-runs-hackthon .

run:
	python src/pipeline.py

test:
	pytest tests/ -v

# The Titanium Cage (Game Day Simulation)
gameday:
	docker run --rm --memory="16g" --memory-swap="16g" --cpus="1.0" india-runs-hackthon