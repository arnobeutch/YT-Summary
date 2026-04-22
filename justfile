default:
    @just --list

lint:
    uv run ruff check .
    uv run ruff format --check .

format:
    uv run ruff format .
    uv run ruff check --fix .

typecheck:
    uv run pyright

test:
    uv run pytest

all: lint typecheck test
