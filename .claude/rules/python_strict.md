# Python strict quality — non-negotiable

This project targets **Python 3.13+** and runs under **strict Pylance / pyright** and **ruff `select = ["ALL"]`**. The editor's Problems panel is ground truth. The `pyright` CLI with no config uses relaxed defaults and will lie; this project's `pyproject.toml` pins strict mode so CLI and editor agree.

After every `Edit` / `Write` on a `.py` file, `.claude/hooks/python-quality.sh` runs `ruff check --fix`, `ruff format`, and `pyright --outputjson`. On any error it exits 2 and pipes the diagnostics to stderr — **read them and fix before moving on**. Don't declare the change done until the hook is silent.

## Core conventions

- Python 3.13+: `X | Y` unions, `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`.
- Dataclasses for models, `Protocol` for interfaces, `async/await` for I/O.
- Unused imports and unused variables are errors. Don't leave them.
- No `# type: ignore` without a one-line explanation of what it's silencing and why.
- No docstrings on obvious methods. Comments only where logic isn't self-evident.

## The five recurring strict-Pylance failure modes

Cases where `x: Any = ...` alone is *not* enough — `reportUnknownVariableType` still fires.

### 1. `json.load` / `yaml.safe_load` / any untyped external return

Assign to an explicit `Any`, then `cast` or validate-then-cast:

```python
from typing import Any, cast

loaded: Any = json.load(fh)
raw = cast(dict[str, Any], loaded) if isinstance(loaded, dict) else {}
```

Never pass the raw result straight into a typed function — it propagates Unknown.

### 2. Untyped C extensions (`pypff`, `libxml2`, etc.)

Even when stubbed as `Any`, attribute returns still fire `reportUnknownVariableType`. `cast(Any, ...)` at every call site, and annotate loop variables:

```python
from typing import Any, cast

handle: Any = cast(Any, ext.open(path))
records: list[Any] = list(getattr(handle, "records", None) or [])
for rec in records:  # rec is Any — OK
    ...
```

Bare `x: Any = untyped_call()` without the explicit `cast` still trips strict rules.

### 3. `tqdm` wrapping an iterator

`tqdm`'s stubs return `tqdm[NoReturn]`. Reassigning breaks inference. Use a separate typed variable:

```python
from collections.abc import Iterable, Iterator

iter_in: Iterator[T] = source_iter
iter_out: Iterable[T] = tqdm(iter_in, desc="...") if show_progress else iter_in
```

### 4. Generics must be parameterized

`dict` → `dict[str, Any]`, `list` → `list[SomeType]`. Bare generics in annotations trip `reportMissingTypeArgument`. Same for return types.

### 5. Dataclass `default_factory` must be parameterized

A bare `list` / `dict` as factory yields `list[Unknown]` / `dict[Unknown, Unknown]` under strict Pylance — even when the field annotation is explicit. Use the subscripted generic as the factory itself (runtime-callable since Python 3.9):

```python
from dataclasses import field

items: list[str] = field(default_factory=list[str])
index: dict[str, int] = field(default_factory=dict[str, int])
```

A `lambda: [...]` literal with a uniform element type is also fine — Pylance infers the element type from the literal.

## Bonus: `d.get("k", [])` / `d.get("k", {})` with bare defaults

On a `dict[str, Any]`, `.get("k", [])` returns `Any | list[Unknown]`. The `list[Unknown]` taints any container it enters and cascades. Strip the Unknown at the source:

```python
signals = cast(list[Any], d.get("signals") or [])
probative = cast(dict[str, Any], d.get("probative_value") or {})
```

Also: annotate dict/list literals before appending (`entry: dict[str, Any] = {...}; container.append(entry)`) — inline `.append({...})` doesn't always pick up the container's parameterization.

## Tests

Strict rules apply in `tests/` too. Only annotation boilerplate (`ANN`, `INP001`, `ARG`) and test-specific overrides (`S101`, `PLR2004`) are relaxed via per-file-ignores — correctness rules (`F*`, `B*`, `S*` beyond `S101`/`S105`/`S106`/`S108`) stay on.

## Decision protocol for ruff + pyright findings

Every diagnostic the hook surfaces gets a judgment, regardless of severity (ruff violation, pyright error, pyright warning). No "it's just a warning" pass-throughs. Three outcomes, from most local to most global:

1. **Fix the code.** First choice when the rule flags a real improvement (`PTH123`, `UP045`, `FBT001`, `ANN204`…) or a real bug. Prefer semantic fixes over suppressions — e.g. `hashlib.sha1(data, usedforsecurity=False)` beats `# noqa: S324`.
2. **Local suppression.** Either `# noqa: XXX  # <reason>` (ruff) or `# type: ignore[code]  # <reason>` (pyright) at the site, or a `[tool.ruff.lint.per-file-ignores]` entry in *this project's* `pyproject.toml` only. Use when the rule is right in general but wrong in this specific context (`BLE001` at a C-extension boundary, `C901` on irreducible domain logic). Does **not** propagate to Boilerplate.
3. **Global ignore.** Add to `[tool.ruff.lint] ignore` (or flip a pyright `report*` to `false`). Use only when the rule is wrong *in principle* for our style. Triggers the full sync protocol (see below).

**Bias toward local.** A rule firing at a handful of context-bound sites is a local decision even if it could fire elsewhere later. Global ignores are reserved for rules that don't fit the project style in principle. When in doubt, stay local — you can always promote a recurring local suppression to global later.

**Claude's role:** for each distinct rule code, propose an outcome with rationale; the user confirms or redirects before any edit.

## Ignore decisions (append-only)

Rationale for each entry in `pyproject.toml`'s `[tool.ruff.lint] ignore`. New additions get a dated inline comment in `pyproject.toml` AND a line here. Rules kept enforced (that a human might assume are ignored) are also worth recording.

| Rule | Status | Date | Rationale |
|------|--------|------|-----------|
| `E501` / `COM812` / `ISC001` | ignored | 2026-04-16 | Formatter owns line length + trailing commas; these conflict with `ruff format`. |
| `D` (pydocstyle) | ignored | 2026-04-16 | No forced docstrings on internal code. Comments only where logic isn't self-evident. |
| `CPY` | ignored | 2026-04-16 | No copyright headers. |
| `ANN401` | ignored | 2026-04-16 | Explicit `Any` is required at untyped boundaries (json.load, C extensions). |
| `FIX`, `TD002`, `TD003` | ignored | 2026-04-16 | Inline fixme/todo comments are fine; no author/issue mandate. |
| `TRY003` | ignored | 2026-04-16 | Long messages in exception constructors are fine. |
| `PLR0913` | ignored | 2026-04-16 | "Too many arguments" is an opinion, not a bug. |
| `PLR2004` | ignored | 2026-04-16 | Magic-number rule too noisy; `tests/**` per-file override handles the common case. |
| `TC001` / `TC002` / `TC003` | ignored | 2026-04-16 | No perf-heavy code here; runtime-visible imports preferred over `TYPE_CHECKING` churn. |
| `TC006` | ignored | 2026-04-16 | `cast(T, x)` → `cast("T", x)` is pointless with `from __future__ import annotations` active; micro-perf at readability's expense. |
| `EM101` / `EM102` | ignored | 2026-04-16 | Inline `raise X(f"...")` is fine; no mandatory `msg = ...; raise X(msg)` dance. |
| `TID252` | ignored | 2026-04-16 | Relative imports (`from .base`, `from ..models`) are idiomatic inside a package. |
| `C901` / `PLR0911` / `PLR0912` / `PLR0915` | ignored | 2026-04-16 | Complexity metrics are opinion, not bug (same family as `PLR0913`). |
| `ERA001`, `G004`, `PIE808`, `T201`, `S311` | ignored | pre-baseline | Inherited defaults (print allowed, f-string logging fine, etc.). |
| `PTH123` | **kept** | 2026-04-16 | `open(path)` vs `path.open()` matters — enforce even when `path` is a `pathlib.Path`. |
| `PTH117` | **kept** | 2026-04-16 | `os.path.isabs(p)` → `Path(p).is_absolute()`. Same family as PTH123. |
| `FBT001` / `FBT002` / `FBT003` | **kept** | 2026-04-16 | Boolean args must be keyword-only. Call sites read `func(x, flag=True)` instead of `func(x, True)` — worth the mechanical cost. |

## Workflow for evolving this list

1. Add to `ignore = [...]` in `pyproject.toml` with a `# YYYY-MM-DD: <reason>` inline comment.
2. Add one line to the table above.
3. Mirror the change to the other project's `pyproject.toml` AND (for Boilerplate) `bootstrap.py`'s `PYPROJECT_TEMPLATE`.

Never edit only one of the three — drift is the failure mode.
