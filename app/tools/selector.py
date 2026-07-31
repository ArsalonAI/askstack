"""Which tool definitions enter `tools[]` — TRD §7.3, ablation axis C.

Three implementations behind one protocol. This module decides *which* tools
the model can see; `app/tools/registry.py` owns what they do. The split is the
whole ablation: swapping the selector changes prompt size and selection
accuracy without touching a single tool implementation.

`full` is the control arm and lands first — it needs no tool embeddings, so the
agent loop can be built and demonstrated before the semantic path exists. The
thesis is that `semantic` matches it on accuracy at a fraction of the prompt,
and a control arm you cannot run is not a comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.interfaces import ToolDef


def to_api_tools(tools: Sequence[ToolDef], *, deferred: Sequence[str] = ()) -> list[dict]:
    """`ToolDef` → the Anthropic `tools[]` wire shape.

    Sorted by name, always (§10). The list is built from a vector query in
    `semantic` mode and from a dict elsewhere; either can reorder between runs,
    and a reordered prefix invalidates the prompt cache with no symptom other
    than the bill.
    """
    deferred_names = set(deferred)
    payload = []
    for tool in sorted(tools, key=lambda t: t.name):
        entry: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        if tool.name in deferred_names:
            entry["defer_loading"] = True
        payload.append(entry)
    return payload


class FullToolSelector:
    """Every definition, every request — §7.3's control arm.

    Deliberately the most expensive arm in prompt tokens and the baseline the
    other two have to beat. It is also the only arm whose tool set is constant,
    so it caches perfectly (§9) — which is worth stating plainly, because it
    means `semantic` is not strictly cheaper and the measurement has to show
    where the crossover is.
    """

    mode = "full"

    def __init__(self, catalog: Sequence[ToolDef]) -> None:
        self._catalog = tuple(catalog)

    async def select(self, query: str, k: int) -> list[ToolDef]:
        # Neither argument is used, and that is the point of the arm.
        return sorted(self._catalog, key=lambda t: t.name)

    def extra_request_params(self) -> dict:
        return {}
