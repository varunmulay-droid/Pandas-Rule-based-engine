"""
Pandas Rule-Based Engine
------------------------
Consumes a list of rule dicts (typically produced upstream from the
preprocessed file / a rules config) and applies them to a DataFrame
using vectorized evaluation. No row-by-row Python loops are used for
rule application — only `.query()` per rule (fast, safe expression
evaluation) or `numpy.select` (fastest, single pass) depending on the
"strategy" chosen.

Rule schema (each dict):
    {
        "name": str,               # unique rule identifier
        "condition": str,          # a pandas.DataFrame.query()-compatible expression
        "action": str,             # label / action to assign on match
        "priority": int (optional) # lower = evaluated first; default = list order
    }

Design notes (industry-standard-ish):
- Rules are validated before execution (schema + column existence +
  syntax check) so a bad rule fails loudly, not silently at row 40000.
- "First match wins" vs "last match wins" is configurable — default
  is last-match-wins to match the reference examples (later rules can
  override earlier ones), but this is explicit, not implicit.
- All engine actions are logged via the standard `logging` module.
- `np.select` path is offered for the mutually-exclusive / large-data
  case; `.query()` path is offered for the general case (rules may
  overlap, expressions can reference any valid pandas query syntax).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("rule_engine")


class MatchStrategy(str, Enum):
    FIRST_MATCH = "first_match"   # first rule that matches wins, later rules ignored for that row
    LAST_MATCH = "last_match"     # rules applied in order, later matches overwrite earlier ones


class EngineBackend(str, Enum):
    QUERY = "query"        # df.query() per rule — flexible, supports arbitrary pandas expressions
    NP_SELECT = "np_select"  # numpy.select — fastest, requires mutually exclusive boolean masks


class RuleValidationError(ValueError):
    """Raised when a rule fails schema or safety validation."""


@dataclass
class Rule:
    name: str
    condition: str
    action: str
    priority: int = 0

    def validate(self, columns: list[str]) -> None:
        if not self.name or not isinstance(self.name, str):
            raise RuleValidationError("Rule 'name' must be a non-empty string.")
        if not self.condition or not isinstance(self.condition, str):
            raise RuleValidationError(f"Rule '{self.name}': 'condition' must be a non-empty string.")
        if not self.action or not isinstance(self.action, str):
            raise RuleValidationError(f"Rule '{self.name}': 'action' must be a non-empty string.")
        # Referenced-column sanity check (best-effort: pulls identifier tokens
        # out of the condition and checks they exist somewhere in the frame,
        # or are the pandas special names True/False/and/or/not/etc.)
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", self.condition))
        reserved = {
            "and", "or", "not", "in", "True", "False", "None",
            "abs", "sqrt", "log", "isnull", "notnull",
        }
        unknown = tokens - reserved - set(columns)
        # Only warn (not raise) since query() supports functions/attrs we can't
        # exhaustively enumerate here — but this catches typo'd column names early.
        if unknown:
            logger.warning(
                "Rule '%s' references identifiers not found in DataFrame columns: %s",
                self.name, sorted(unknown),
            )


@dataclass
class RuleEngineConfig:
    strategy: MatchStrategy = MatchStrategy.LAST_MATCH
    backend: EngineBackend = EngineBackend.QUERY
    default_result: str = "Unknown"
    result_column: str = "result"
    matched_rule_column: str | None = "matched_rule"  # set to None to skip tracking which rule fired


class RuleEngine:
    """
    Applies a list of business rules to a DataFrame using vectorized
    pandas/numpy operations (no iterrows()).
    """

    def __init__(self, rules: list[dict[str, Any]], config: RuleEngineConfig | None = None):
        self.config = config or RuleEngineConfig()
        self.rules: list[Rule] = [Rule(**r) for r in rules]
        if not self.rules:
            raise RuleValidationError("At least one rule is required.")
        # Deterministic ordering: explicit priority first, else list order.
        self.rules.sort(key=lambda r: r.priority)

    def _validate_against(self, df: pd.DataFrame) -> None:
        columns = list(df.columns)
        seen_names = set()
        for rule in self.rules:
            rule.validate(columns)
            if rule.name in seen_names:
                raise RuleValidationError(f"Duplicate rule name: '{rule.name}'")
            seen_names.add(rule.name)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_against(df)
        if self.config.backend == EngineBackend.QUERY:
            return self._run_query(df)
        return self._run_np_select(df)

    # ---- Backend 1: df.query() ---------------------------------------
    def _run_query(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.config.result_column] = self.config.default_result
        if self.config.matched_rule_column:
            out[self.config.matched_rule_column] = pd.NA

        ordered = self.rules if self.config.strategy == MatchStrategy.LAST_MATCH else list(reversed(self.rules))
        already_matched = pd.Series(False, index=out.index)

        for rule in ordered:
            try:
                matched_idx = out.query(rule.condition, engine="python").index
            except Exception as exc:  # malformed expression, missing column at runtime, etc.
                logger.error("Rule '%s' failed to evaluate: %s", rule.name, exc)
                raise RuleValidationError(f"Rule '{rule.name}' failed to evaluate: {exc}") from exc

            if self.config.strategy == MatchStrategy.FIRST_MATCH:
                matched_idx = matched_idx.difference(out.index[already_matched])
                already_matched.loc[matched_idx] = True

            out.loc[matched_idx, self.config.result_column] = rule.action
            if self.config.matched_rule_column:
                out.loc[matched_idx, self.config.matched_rule_column] = rule.name

            logger.info("Rule '%s' matched %d rows.", rule.name, len(matched_idx))

        return out

    # ---- Backend 2: numpy.select --------------------------------------
    def _run_np_select(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Requires rules whose conditions can each be evaluated to a boolean
        mask independently (mutually exclusive is recommended, though
        np.select applies the FIRST matching condition per row regardless).
        """
        out = df.copy()
        masks = []
        choices = []
        for rule in self.rules:
            try:
                mask = out.eval(rule.condition, engine="python")
            except Exception as exc:
                logger.error("Rule '%s' failed to evaluate: %s", rule.name, exc)
                raise RuleValidationError(f"Rule '{rule.name}' failed to evaluate: {exc}") from exc
            if mask.dtype != bool:
                raise RuleValidationError(f"Rule '{rule.name}' condition did not evaluate to a boolean mask.")
            masks.append(mask.to_numpy())
            choices.append(rule.action)

        out[self.config.result_column] = np.select(masks, choices, default=self.config.default_result)

        if self.config.matched_rule_column:
            rule_names = [r.name for r in self.rules]
            out[self.config.matched_rule_column] = np.select(masks, rule_names, default=None)

        for rule, mask in zip(self.rules, masks):
            logger.info("Rule '%s' matched %d rows.", rule.name, int(mask.sum()))

        return out


def build_engine_from_config(rules: list[dict[str, Any]], **config_overrides: Any) -> RuleEngine:
    """Convenience factory — e.g. build_engine_from_config(rules, backend='np_select')."""
    cfg = RuleEngineConfig(**config_overrides) if config_overrides else RuleEngineConfig()
    return RuleEngine(rules, cfg)
