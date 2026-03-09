.PHONY: reproduce sync-metrics test lint

reproduce:
	python -m src.train --run-evaluation

sync-metrics:
	python -m src.evaluate

test:
	pytest -q

lint:
	black src tests app
