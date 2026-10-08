.PHONY: setup data lint test clean

setup:
	python -m venv house_venv
	. house_venv/bin/activate && pip install -r requirements.txt && pip install -e .

data:
	python -m house_price.data

lint:
	ruff check src tests

test:
	pytest

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__