.PHONY: reproduce test lint

reproduce:
	python -m src.train --run-evaluation

test:
	pytest -q

lint:
	black src tests app
