"""
Theme-related menu rules.

Four rules that together enforce the weekday → cuisine-theme mapping:

* :class:`ThemeDayMenuRule` — hard constraint: Monday 'mix' day requires
  at least one south-cuisine and one north-cuisine item.
* :class:`ThemeSlotFilterRule` — pre-filter: on chinese / biryani /
  south / north days, narrow each slot's pool to items that fit the
  day's theme.
* :class:`ThemeStarterPreferenceRule` — soft bonus for starters that
  match the day's theme.
* :class:`ThemeFallbackPenaltyRule` — soft penalty for non-theme items
  chosen in starter / veg_dry slots when a theme item was available.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Set

import pandas as pd
from ortools.sat.python import cp_model

from src.constants import BASE_SLOT_NAMES, EXEMPT_FROM_CUISINE

from ..preprocessor.column_mapper import _norm_str, _to_bool01
from .base_menu_rule import (
    BaseMenuRule,
    Diagnostic,
    DiagnosticPhase,
    DiagnosticSeverity,
    DiagnoseContext,
    MenuRuleType,
    MenuRuleSeverity,
)


# ---------------------------------------------------------------------------
# ThemeDayMenuRule
# ---------------------------------------------------------------------------


class ThemeDayMenuRule(BaseMenuRule):
    """
    Enforces Monday mix constraint: >= 1 south + >= 1 north item.

    Config:
    {
        "type": "theme_day",
        "name": "monday_mix"
    }
    """

    def __init__(self, rule_config: Dict[str, Any]):
        super().__init__(rule_config)
        self.rule_type = MenuRuleType.THEME_DAY

    def apply(self, model: cp_model.CpModel, variables: Dict[str, Any],
              menu_data: Any, context: Dict[str, Any]) -> None:
        day_types = context.get('day_types', [])
        south_lits = context.get('monday_south_lits', [])
        north_lits = context.get('monday_north_lits', [])

        if any(dt_ == 'mix' for dt_ in day_types):
            if south_lits:
                model.Add(sum(south_lits) >= 1)
            if north_lits:
                model.Add(sum(north_lits) >= 1)

    def diagnose(self, ctx: DiagnoseContext) -> List[Diagnostic]:
        """For every mix-themed day, verify the non-exempt slot pools
        carry at least one south_indian AND one north_indian item.

        The CP-SAT constraint is ``sum(south_lits) >= 1`` and
        ``sum(north_lits) >= 1``, so a 0-count in either direction is
        a guaranteed infeasibility — emit ERROR.
        """
        diags: List[Diagnostic] = []
        if not any(t == 'mix' for t in ctx.day_types.values()):
            return diags

        cuisine_col = ctx.cfg.cuisine_col if ctx.cfg else 'cuisine_family'
        south_val = ctx.cfg.cuisine_south_value if ctx.cfg else 'south_indian'
        north_val = ctx.cfg.cuisine_north_value if ctx.cfg else 'north_indian'
        base_slots = ctx.active_base_slots or list(BASE_SLOT_NAMES)

        for d in ctx.dates:
            if ctx.day_types.get(d) != 'mix':
                continue
            day_label = d.strftime('%A %d %b')
            south_total = north_total = 0

            for base in base_slots:
                if base in EXEMPT_FROM_CUISINE:
                    continue
                if (d, base) in ctx.skip_cells:
                    continue
                pool = ctx.pools.get(base)
                if pool is None or len(pool) == 0:
                    continue
                if cuisine_col not in pool.columns:
                    continue
                cuisines = pool[cuisine_col].map(_norm_str)
                south_total += int((cuisines == south_val).sum())
                north_total += int((cuisines == north_val).sum())

            for label, count, target_val in (
                ('south_indian', south_total, south_val),
                ('north_indian', north_total, north_val),
            ):
                if count == 0:
                    diags.append(Diagnostic(
                        rule=self.name,
                        rule_type=self.rule_type.value,
                        severity=DiagnosticSeverity.ERROR,
                        phase=DiagnosticPhase.APPLY,
                        message=(
                            f"Mix theme on {day_label} requires ≥1 "
                            f"{label} item across non-exempt slots, "
                            f"but the pools have 0."
                        ),
                        suggestion=(
                            f"Add at least one {label} item to a "
                            f"non-exempt slot, or change {day_label}'s "
                            f"theme in the customisation editor."
                        ),
                        affected={
                            'date': d.isoformat(),
                            'day_type': 'mix',
                            'cuisine': target_val,
                            'count': 0,
                        },
                    ))
                elif count == 1:
                    diags.append(Diagnostic(
                        rule=self.name,
                        rule_type=self.rule_type.value,
                        severity=DiagnosticSeverity.WARNING,
                        phase=DiagnosticPhase.APPLY,
                        message=(
                            f"Mix theme on {day_label}: only 1 {label} "
                            f"item available across non-exempt slots; "
                            f"any cooldown / theme filter that drops it "
                            f"will make this day infeasible."
                        ),
                        suggestion=f"Add more {label} items to the ontology.",
                        affected={
                            'date': d.isoformat(),
                            'day_type': 'mix',
                            'cuisine': target_val,
                            'count': 1,
                        },
                    ))
        return diags


# ---------------------------------------------------------------------------
# ThemeSlotFilterRule
# ---------------------------------------------------------------------------

# Slots that get Chinese-specific filtering
_CHINESE_FLAG_MAP = {
    'rice': 'is_chinese_fried_rice',
    'veg_gravy': 'is_chinese_veg_gravy',
    'nonveg_main': 'is_chinese_chicken_gravy',
}

# Biryani flag map
_BIRYANI_FLAG_MAP = {
    'rice': 'is_mixedveg_biryani',
    'nonveg_main': 'is_nonveg_biryani',
}


def _chinese_side_mask(pool: pd.DataFrame) -> pd.Series:
    """Detect Chinese-appropriate veg_dry items via text heuristics."""
    text = (pool['item'].astype(str) + ' ' +
            pool.get('sub_category', pd.Series('', index=pool.index)).astype(str))
    text = text.str.lower()
    return (
        text.str.contains('chinese', na=False) |
        text.str.contains('manchurian', na=False) |
        text.str.contains('schezwan', na=False) |
        text.str.contains('szechuan', na=False) |
        text.str.contains('gobi_65', na=False) |
        text.str.contains('gobi 65', na=False) |
        text.str.contains('baby_corn', na=False) |
        text.str.contains('baby corn', na=False) |
        text.str.contains('noodle', na=False) |
        text.str.contains('chilli', na=False)
    )


class ThemeSlotFilterRule(BaseMenuRule):
    """
    Config:
    {
        "type": "theme_slot_filter",
        "name": "theme_cuisine_filter",
        "exempt_slots": ["welcome_drink", "dal", "sambar", "rasam",
                         "starter", "soup", "salad", "healthy_rice"]
    }
    """

    def __init__(self, rule_config: Dict[str, Any]):
        super().__init__(rule_config)
        self.rule_type = MenuRuleType.THEME_SLOT_FILTER
        exempt = rule_config.get('exempt_slots')
        self.exempt_slots: Set[str] = set(exempt) if exempt else set(EXEMPT_FROM_CUISINE)

    def pre_filter_pool(self, pool: pd.DataFrame, date: dt.date,
                        base_slot: str, day_type: str,
                        filter_context: Dict[str, Any]) -> pd.DataFrame:
        if len(pool) == 0:
            return pool

        cfg = filter_context.get('cfg')

        if day_type == 'chinese':
            return self._filter_chinese(pool, base_slot, cfg)
        if day_type == 'biryani':
            return self._filter_biryani(pool, base_slot, cfg)
        if day_type in ('south', 'north'):
            return self._filter_cuisine(pool, base_slot, day_type, cfg)
        # 'mix', 'holiday', 'normal' — no theme filtering
        return pool

    def _filter_chinese(self, pool: pd.DataFrame, base_slot: str, cfg) -> pd.DataFrame:
        flag_col = _CHINESE_FLAG_MAP.get(base_slot)
        if flag_col and flag_col in pool.columns:
            filtered = pool[pool[flag_col].map(_to_bool01) == 1]
            if len(filtered) > 0:
                return filtered

        if base_slot == 'veg_dry':
            mask = _chinese_side_mask(pool)
            filtered = pool[mask]
            if len(filtered) > 0:
                return filtered

        # Exempt slots and slots without flags: return unfiltered
        return pool

    def _filter_biryani(self, pool: pd.DataFrame, base_slot: str, cfg) -> pd.DataFrame:
        flag_col = _BIRYANI_FLAG_MAP.get(base_slot)
        if flag_col and flag_col in pool.columns:
            filtered = pool[pool[flag_col].map(_to_bool01) == 1]
            if len(filtered) > 0:
                return filtered
        return pool

    def _filter_cuisine(self, pool: pd.DataFrame, base_slot: str,
                        day_type: str, cfg) -> pd.DataFrame:
        cuisine_col = cfg.cuisine_col if cfg else 'cuisine_family'
        south_val = cfg.cuisine_south_value if cfg else 'south_indian'
        north_val = cfg.cuisine_north_value if cfg else 'north_indian'

        target = south_val if day_type == 'south' else north_val

        # Bread cuisine lock: south bread on south days, non-south on others
        if base_slot == 'bread':
            if cuisine_col in pool.columns:
                if day_type == 'south':
                    filtered = pool[pool[cuisine_col].map(_norm_str) == south_val]
                else:
                    filtered = pool[pool[cuisine_col].map(_norm_str) != south_val]
                if len(filtered) > 0:
                    return filtered
            return pool

        # Exempt slots: no cuisine filtering
        if base_slot in self.exempt_slots:
            return pool

        # Non-exempt slots: filter by matching cuisine_family
        if cuisine_col in pool.columns:
            filtered = pool[pool[cuisine_col].map(_norm_str) == target]
            if len(filtered) > 0:
                return filtered

        return pool

    def apply(self, model: cp_model.CpModel, variables: Dict[str, Any],
              menu_data: Any, context: Dict[str, Any]) -> None:
        pass  # All filtering happens in pre_filter_pool

    def diagnose(self, ctx: DiagnoseContext) -> List[Diagnostic]:
        """Project the theme slot filter for every (date, slot) on a
        themed day and report:

          - WARNING when the configured flag column doesn't match any
            items at all (filter would empty the pool; the rule itself
            falls back to unfiltered, so the user gets a non-theme
            menu silently — surfacing this lets them fix the data).
          - INFO   when the filter narrows the pool by ≥50%.

        Never ERROR: this rule's design is to fall back to the
        unfiltered pool when filtering would empty it, so it can't be
        the *cause* of an infeasibility on its own. (The downstream
        cuisine/coupling/item_cooldown rules emit their own errors
        based on the data the user actually has.)
        """
        diags: List[Diagnostic] = []
        cfg = ctx.cfg
        cuisine_col = cfg.cuisine_col if cfg else 'cuisine_family'
        south_val = cfg.cuisine_south_value if cfg else 'south_indian'
        north_val = cfg.cuisine_north_value if cfg else 'north_indian'

        base_slots = ctx.active_base_slots or list(BASE_SLOT_NAMES)

        for d in ctx.dates:
            day_type = ctx.day_types.get(d, '')
            if day_type not in ('chinese', 'biryani', 'south', 'north'):
                continue
            day_label = d.strftime('%A %d %b')

            for base in base_slots:
                if (d, base) in ctx.skip_cells:
                    continue
                if base in self.exempt_slots and base != 'bread':
                    continue
                pool = ctx.pools.get(base)
                if pool is None or len(pool) == 0:
                    continue

                filtered_size = self._project_filter_size(
                    pool, base, day_type, cuisine_col, south_val, north_val,
                )
                if filtered_size is None:
                    continue  # No filter applies to this (slot, day_type)
                slot_label = base.replace('_', ' ')

                if filtered_size == 0:
                    diags.append(Diagnostic(
                        rule=self.name,
                        rule_type=self.rule_type.value,
                        severity=DiagnosticSeverity.WARNING,
                        phase=DiagnosticPhase.PRE_FILTER,
                        message=(
                            f"{day_type.capitalize()} {day_label}: 0 items "
                            f"match the {day_type} filter for the "
                            f"{slot_label} slot. Falling back to the "
                            f"unfiltered pool — the plan will use a "
                            f"non-{day_type} item here."
                        ),
                        suggestion=(
                            f"Tag at least one {slot_label} item as "
                            f"{day_type} in the ontology, or accept "
                            f"the non-theme fallback."
                        ),
                        affected={
                            'date': d.isoformat(),
                            'slot': base,
                            'day_type': day_type,
                            'pool_size_before': len(pool),
                            'pool_size_after': 0,
                        },
                    ))
                elif filtered_size < len(pool) // 2:
                    diags.append(Diagnostic(
                        rule=self.name,
                        rule_type=self.rule_type.value,
                        severity=DiagnosticSeverity.INFO,
                        phase=DiagnosticPhase.PRE_FILTER,
                        message=(
                            f"{day_type.capitalize()} {day_label}: filter "
                            f"narrowed {slot_label} pool from {len(pool)} "
                            f"to {filtered_size} items."
                        ),
                        suggestion="No action needed.",
                        affected={
                            'date': d.isoformat(),
                            'slot': base,
                            'day_type': day_type,
                            'pool_size_before': len(pool),
                            'pool_size_after': filtered_size,
                        },
                    ))
        return diags

    def _project_filter_size(
        self,
        pool: pd.DataFrame,
        base_slot: str,
        day_type: str,
        cuisine_col: str,
        south_val: str,
        north_val: str,
    ):
        """Return the post-filter pool size for *(base_slot, day_type)*
        WITHOUT applying the rule's fallback-to-unfiltered behaviour.

        Returns ``None`` when no filter applies to this combination so
        the caller can skip it.
        """
        if day_type == 'chinese':
            flag_col = _CHINESE_FLAG_MAP.get(base_slot)
            if flag_col and flag_col in pool.columns:
                return int((pool[flag_col].map(_to_bool01) == 1).sum())
            if base_slot == 'veg_dry':
                return int(_chinese_side_mask(pool).sum())
            return None

        if day_type == 'biryani':
            flag_col = _BIRYANI_FLAG_MAP.get(base_slot)
            if flag_col and flag_col in pool.columns:
                return int((pool[flag_col].map(_to_bool01) == 1).sum())
            return None

        # south / north
        if base_slot == 'bread' and cuisine_col in pool.columns:
            cuisines = pool[cuisine_col].map(_norm_str)
            if day_type == 'south':
                return int((cuisines == south_val).sum())
            return int((cuisines != south_val).sum())

        if base_slot in self.exempt_slots:
            return None  # No cuisine filter on exempt non-bread slots

        if cuisine_col not in pool.columns:
            return None
        target = south_val if day_type == 'south' else north_val
        return int((pool[cuisine_col].map(_norm_str) == target).sum())


# ---------------------------------------------------------------------------
# ThemeStarterPreferenceRule
# ---------------------------------------------------------------------------


class ThemeStarterPreferenceRule(BaseMenuRule):
    """
    Config:
    {
        "type": "theme_starter_preference",
        "name": "prefer_theme_starters",
        "bonus_weight": 1000000
    }
    """

    severity = MenuRuleSeverity.SOFT

    def __init__(self, rule_config: Dict[str, Any]):
        super().__init__(rule_config)
        self.rule_type = MenuRuleType.THEME_STARTER_PREFERENCE
        self.bonus_weight = rule_config.get('bonus_weight', 1000000)

    def apply(self, model: cp_model.CpModel, variables: Dict[str, Any],
              menu_data: Any, context: Dict[str, Any]) -> None:
        pass  # This rule contributes to objective only

    def get_objective_terms(self, model: cp_model.CpModel,
                            context: Dict[str, Any]) -> List:
        cells = context.get('cells', [])
        dates = context.get('dates', [])
        find_cells = context.get('find_cells_fn')
        link_any = context.get('link_any_fn')
        cfg = context.get('cfg')

        if not find_cells or not link_any or not cfg or not cfg.prefer_theme_starter:
            return []

        ok_vars = []
        for di in range(len(dates)):
            for idx, scell in enumerate(find_cells(cells, di, 'starter'), start=1):
                lits = [v for v, pref in zip(scell.x_vars, scell.theme_pref_flags) if pref]
                if lits:
                    ok = model.NewBoolVar(f'starter_theme_ok_{di}_{idx}')
                    link_any(model, lits, ok)
                    ok_vars.append(ok)

        if ok_vars:
            return [sum(ok_vars) * self.bonus_weight]
        return []


# ---------------------------------------------------------------------------
# ThemeFallbackPenaltyRule
# ---------------------------------------------------------------------------


class ThemeFallbackPenaltyRule(BaseMenuRule):
    """
    Config:
    {
        "type": "theme_fallback_penalty",
        "name": "penalize_non_theme_fallback",
        "penalty": 2000000
    }
    """

    severity = MenuRuleSeverity.SOFT

    def __init__(self, rule_config: Dict[str, Any]):
        super().__init__(rule_config)
        self.rule_type = MenuRuleType.THEME_FALLBACK_PENALTY
        self.penalty = rule_config.get('penalty', 2000000)

    def apply(self, model: cp_model.CpModel, variables: Dict[str, Any],
              menu_data: Any, context: Dict[str, Any]) -> None:
        return

    def get_objective_terms(self, model: cp_model.CpModel,
                            context: Dict[str, Any]) -> List:
        fallback_bools = context.get('theme_fallback_bools') or []
        if not fallback_bools:
            return []
        return [sum(fallback_bools) * (-abs(int(self.penalty)))]
