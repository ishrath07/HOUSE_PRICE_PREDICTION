.PHONY: setup data train evaluate predict lint format test clean

ifeq ($(OS),Windows_NT)
VENV_PYTHON = .venv/Scripts/python.exe
else
VENV_PYTHON = .venv/bin/python
endif

setup:
	python -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
	$(VENV_PYTHON) -m pip install -e .

data:
	$(VENV_PYTHON) -m house_price.data

train:
	$(VENV_PYTHON) -m house_price.train

evaluate:
	$(VENV_PYTHON) -m house_price.evaluate

predict:
	$(VENV_PYTHON) -m house_price.predict $(ARGS)

lint:
	$(VENV_PYTHON) -m ruff check src tests

format:
	$(VENV_PYTHON) -m black src tests

test:
	$(VENV_PYTHON) -m pytest

clean:
	$(VENV_PYTHON) -c "import shutil; from pathlib import Path; [shutil.rmtree(p, ignore_errors=True) for p in (Path('.pytest_cache'), Path('.ruff_cache'))]"
