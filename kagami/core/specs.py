"""Base policy-spec grammar and parsing for Kagami inference engines.

Canonical grammar::

    kind1[@kind2[...]] = locator1[@locator2[...]] [?key=val&...]

Examples::

    marvel=/path/to/ckpt
    ante=wandb:entity/proj/run
    marvel@yomi=policy_ckpt@belief_ckpt?action_temp=0.5
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qsl


@dataclass(frozen = True)
class PolicySpec:
    """Base policy spec produced by :class:`PolicySpecParser`.

    Game-specific subclasses add fields like ``action_temp``,
    ``belief_checkpoint``, etc.
    """

    kind: str
    canonical_spec: str
    checkpoint_locators: tuple[str, ...] = ()
    resolved_checkpoints: tuple[str | None, ...] = ()


class PolicySpecParser:
    """Game-agnostic base for the ``kind=locator`` spec grammar.

    Subclasses override :meth:`parse` to produce game-specific
    :class:`PolicySpec` subclass instances.  The helper methods below
    handle the shared mechanics so that subclasses stay concise.
    """

    # ------------------------------------------------------------------
    # Grammar splitting
    # ------------------------------------------------------------------

    @staticmethod
    def split_spec(raw: str) -> tuple[list[str], list[str], dict[str, str]]:
        """Split a raw spec string into (kinds, locators, raw_overrides).

        Returns
        -------
        kinds : list[str]
            ``@``-delimited kind chain (e.g. ``["marvel", "yomi"]``).
        locators : list[str]
            ``@``-delimited checkpoint locator chain, **before** any
            normalization (e.g. ``["./ckpt1", "wandb:e/p/r"]``).
        raw_overrides : dict[str, str]
            Query-string key/value pairs as raw strings.  An empty dict
            when no ``?`` segment is present.
        """
        spec = str(raw).strip()
        if not spec:
            raise ValueError("Empty policy spec.")

        base, has_query, query = spec.partition("?")

        eq_idx = base.find("=")
        if eq_idx < 0:
            raise ValueError(
                f"Policy spec must contain '=' separating kind(s) from locator(s): {raw!r}"
            )

        kind_part = base[:eq_idx]
        locator_part = base[eq_idx + 1 :]

        kinds = kind_part.split("@")
        if not all(k.strip() for k in kinds):
            raise ValueError(f"Empty kind segment in policy spec: {raw!r}")
        kinds = [k.strip() for k in kinds]

        locators = locator_part.split("@")
        locators = [loc.strip() for loc in locators]

        raw_overrides: dict[str, str] = {}
        if has_query:
            raw_overrides = PolicySpecParser._parse_raw_query(query)

        return kinds, locators, raw_overrides

    # ------------------------------------------------------------------
    # Query-string helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_query_overrides(
        query: str,
        *,
        allowed_keys: Sequence[str],
    ) -> dict[str, str]:
        """Parse a query string into validated key/value pairs.

        Raises on unknown keys, duplicates, or empty values.
        Returns raw string values; callers cast to float/int.
        """
        if not query:
            return {}
        raw = PolicySpecParser._parse_raw_query(query)
        allowed = set(allowed_keys)
        for key in raw:
            if key not in allowed:
                choices = ", ".join(allowed_keys)
                raise ValueError(f"Unknown override key {key!r}. Allowed keys: {choices}.")
        return raw

    @staticmethod
    def _parse_raw_query(query: str) -> dict[str, str]:
        try:
            pairs = parse_qsl(query, keep_blank_values = True, strict_parsing = True)
        except ValueError as exc:
            raise ValueError(f"Invalid query override segment: ?{query}") from exc

        seen: set[str] = set()
        parsed: dict[str, str] = {}
        for key, raw_value in pairs:
            if key in seen:
                raise ValueError(f"Duplicate override key: {key!r}.")
            seen.add(key)
            if raw_value.strip() == "":
                raise ValueError(f"Override {key!r} must have a value.")
            parsed[key] = raw_value
        return parsed

    # ------------------------------------------------------------------
    # Checkpoint locator normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_checkpoint_locator(
        raw: str,
        *,
        resolve_base: Path | None = None,
    ) -> str:
        """Normalize a checkpoint locator.

        * Strips surrounding whitespace and quotes.
        * Returns ``wandb:`` locators unchanged.
        * Resolves relative filesystem paths against *resolve_base*
          (defaults to ``Path(".")``).
        """
        locator = str(raw).strip().strip("'").strip('"')
        if not locator:
            raise ValueError("Checkpoint locator must be non-empty.")
        if locator.startswith("wandb:"):
            return locator
        checkpoint = Path(locator).expanduser()
        if not checkpoint.is_absolute():
            base = resolve_base if resolve_base is not None else Path(".")
            checkpoint = (base / checkpoint).resolve()
        return str(checkpoint)

    # ------------------------------------------------------------------
    # Numeric parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_non_negative_float(text: str, *, name: str) -> float:
        try:
            value = float(str(text).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a float, got {text!r}.") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return float(value)

    @staticmethod
    def parse_float(text: str, *, name: str) -> float:
        try:
            value = float(str(text).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a float, got {text!r}.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return float(value)

    @staticmethod
    def parse_positive_int(text: str, *, name: str) -> int:
        try:
            value = int(str(text).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {text!r}.") from exc
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    @staticmethod
    def format_float(value: float) -> str:
        return format(float(value), "g")

    # ------------------------------------------------------------------
    # Directory-safe slug
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize_dir_component(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_").lower()


__all__ = [
    "PolicySpec",
    "PolicySpecParser",
]
