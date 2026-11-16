export UV_LINK_MODE ?= copy
# QueryLens: all commands go through uv. On Windows install GNU make (e.g. `choco install make`)
# or call the same `uv run ...` lines directly.
SIZE ?= small
ROLE ?= analyst
SLICE ?= dev
REPEATS ?= 1
CONFIG ?= b_mistral
RUN ?=
Q ?= net revenue by division in 2026 Q2

.PHONY: setup lint test world world-check dictionary agent viewer eval gate report redteam frontier models-check quota datasets

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

test:
	uv run pytest -q

world:
	uv run python -m bazaar_world.generate --size $(SIZE) --seed 42 --out data/

world-check:
	uv run python -m bazaar_world.check --size $(SIZE)

dictionary:
	uv run python -m bazaar_world.dictionary_check --size $(SIZE)

agent:
	uv run querylens ask "$(Q)" --role $(ROLE) --size $(SIZE) --config $(CONFIG)

viewer:
	uv run streamlit run app/viewer.py

datasets:
	uv run evalkit datasets report

eval:
	uv run evalkit run --slice $(SLICE) --size $(SIZE) --repeats $(REPEATS) --config $(CONFIG)

gate:
	uv run evalkit gate --run $(RUN)

report:
	uv run evalkit report --run $(RUN)

redteam:
	npx --yes promptfoo@latest redteam run -c redteam/promptfooconfig.yaml
	uv run evalkit run --slice adv --size $(SIZE) --repeats 1 --config $(CONFIG)

frontier:
	uv run evalkit frontier --slice $(SLICE) --size $(SIZE)

models-check:
	uv run querylens models-check

quota:
	uv run querylens quota
