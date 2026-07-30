"""Area name -> path globs — TRD §5.5.

Two rules from §5.5 are load-bearing and easy to lose:

* An unrecognised area name is an **error**, never an empty result. "No commits
  in the payments area" and "there is no area called payments" must not look
  the same to the manager.
* Every answer filtered by area **names the globs it resolved**, so a bad
  mapping is visible in the output rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import asyncpg
import yaml


class UnknownArea(KeyError):
    """Raised for an area name that `areas.yaml` does not define."""

    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        self.known = known
        super().__init__(
            f"no area called {name!r}. Defined areas: {', '.join(known) or '(none)'}"
        )


@dataclass(frozen=True)
class Area:
    name: str
    description: str | None
    path_globs: tuple[str, ...]

    def sql_prefixes(self) -> list[str]:
        """Glob -> a `LIKE` prefix pattern.

        `pr_files_path_idx` is a `text_pattern_ops` index, so a prefix `LIKE`
        uses it and a regex would not. Only trailing `**`/`*` are supported,
        which covers every glob shape §5.5 shows.
        """
        prefixes = []
        for glob in self.path_globs:
            prefixes.append(glob.removesuffix("**").removesuffix("*") + "%")
        return prefixes


def load_areas_file(path: Path) -> list[Area]:
    raw = yaml.safe_load(path.read_text()) or []
    areas = [
        Area(
            name=entry["name"],
            description=entry.get("description"),
            path_globs=tuple(entry["path_globs"]),
        )
        for entry in raw
    ]
    names = [a.name for a in areas]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate area names in {path}: {sorted(duplicates)}")
    return areas


async def sync_areas(conn: asyncpg.Connection, areas: list[Area]) -> int:
    """Replace the `areas` table from the file. Curated, so the file wins."""
    await conn.execute("DELETE FROM areas")
    if areas:
        await conn.executemany(
            "INSERT INTO areas (name, path_globs, description) VALUES ($1,$2,$3)",
            [(a.name, list(a.path_globs), a.description) for a in areas],
        )
    return len(areas)


async def resolve(conn: asyncpg.Connection, name: str) -> Area:
    row = await conn.fetchrow(
        "SELECT name, description, path_globs FROM areas WHERE name = $1", name
    )
    if row is None:
        known = [r["name"] for r in await conn.fetch("SELECT name FROM areas ORDER BY name")]
        raise UnknownArea(name, known)
    return Area(row["name"], row["description"], tuple(row["path_globs"]))
